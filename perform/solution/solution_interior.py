import os

import numpy as np
from scipy.sparse import csr_matrix

from perform.constants import REAL_TYPE, RES_NORM_PRIM_DEFAULT
from perform.solution.solution_phys import SolutionPhys


class SolutionInterior(SolutionPhys):
    """
    Solution of interior domain
    """

    def __init__(self, gas, sol_prim_in, solver, num_cells, num_reactions, time_int):
        super().__init__(gas, num_cells, sol_prim_in=sol_prim_in)

        gas = self.gas_model

        if num_reactions > 0:
            self.wf = np.zeros((num_reactions, num_cells), dtype=REAL_TYPE)
        self.source = np.zeros((gas.num_species, num_cells), dtype=REAL_TYPE)
        self.rhs = np.zeros((gas.num_eqs, num_cells), dtype=REAL_TYPE)

        # add bulk velocity and update state if requested
        if solver.vel_add != 0.0:
            self.sol_prim[1, :] += solver.vel_add
            self.update_state(from_cons=False)

        # initializing time history
        self.sol_hist_cons = [self.sol_cons.copy()] * (time_int.time_order + 1)
        self.sol_hist_prim = [self.sol_prim.copy()] * (time_int.time_order + 1)

        # RHS storage for multi-stage schemes
        self.rhs_hist = [self.rhs.copy()] * (time_int.time_order + 1)

        # snapshot storage matrices, store initial condition
        if solver.prim_out:
            self.prim_snap = np.zeros((gas.num_eqs, num_cells, solver.num_snaps + 1), dtype=REAL_TYPE)
            self.prim_snap[:, :, 0] = self.sol_prim.copy()
        if solver.cons_out:
            self.cons_snap = np.zeros((gas.num_eqs, num_cells, solver.num_snaps + 1), dtype=REAL_TYPE)
            self.cons_snap[:, :, 0] = self.sol_cons.copy()

        # these don't include the source/RHS associated with the final solution
        # TODO: calculate at final solution for DEIM stuff
        if solver.source_out:
            self.source_snap = np.zeros((gas.num_species, num_cells, solver.num_snaps), dtype=REAL_TYPE)
        if solver.rhs_out:
            self.rhs_snap = np.zeros((gas.num_eqs, num_cells, solver.num_snaps), dtype=REAL_TYPE)

        if (time_int.time_type == "implicit") or (solver.run_steady):
            # norm normalization constants
            # TODO: will need a normalization constant for
            # 	conservative residual when it's implemented
            if (len(solver.res_norm_prim) == 1) and (solver.res_norm_prim[0] is None):
                solver.res_norm_prim = [None] * gas.num_eqs
            else:
                assert len(solver.res_norm_prim) == gas.num_eqs
            for var_idx in range(gas.num_eqs):
                if solver.res_norm_prim[var_idx] is None:
                    # 0: pressure, 1: velocity, 2: temperature, >=3: species
                    solver.res_norm_prim[var_idx] = RES_NORM_PRIM_DEFAULT[min(var_idx, 3)]

            # residual norm storage
            if time_int.time_type == "implicit":

                self.res = np.zeros((gas.num_eqs, num_cells), dtype=REAL_TYPE)
                self.res_norm_l2 = 0.0
                self.res_norm_l1 = 0.0
                self.res_norm_history = np.zeros((solver.num_steps, 2), dtype=REAL_TYPE)

                if (time_int.dual_time) and (time_int.adapt_dtau):
                    self.srf = np.zeros(num_cells, dtype=REAL_TYPE)

                # CSR matrix indices
                num_elements = gas.num_eqs ** 2 * num_cells
                self.jacob_dim = gas.num_eqs * num_cells

                row_idxs_center = np.zeros(num_elements, dtype=np.int32)
                col_idxs_center = np.zeros(num_elements, dtype=np.int32)
                row_idxs_upper = np.zeros(num_elements - gas.num_eqs ** 2, dtype=np.int32)
                col_idxs_upper = np.zeros(num_elements - gas.num_eqs ** 2, dtype=np.int32)
                row_idxs_lower = np.zeros(num_elements - gas.num_eqs ** 2, dtype=np.int32)
                col_idxs_lower = np.zeros(num_elements - gas.num_eqs ** 2, dtype=np.int32)

                # TODO: definitely a faster way to do this
                lin_idx_A = 0
                lin_idx_B = 0
                lin_idx_C = 0
                for i in range(gas.num_eqs):
                    for j in range(gas.num_eqs):
                        for k in range(num_cells):

                            row_idxs_center[lin_idx_A] = i * num_cells + k
                            col_idxs_center[lin_idx_A] = j * num_cells + k
                            lin_idx_A += 1

                            if k < (num_cells - 1):
                                row_idxs_upper[lin_idx_B] = i * num_cells + k
                                col_idxs_upper[lin_idx_B] = j * num_cells + k + 1
                                lin_idx_B += 1

                            if k > 0:
                                row_idxs_lower[lin_idx_C] = i * num_cells + k
                                col_idxs_lower[lin_idx_C] = j * num_cells + k - 1
                                lin_idx_C += 1

                self.jacob_row_idxs = np.concatenate((row_idxs_center, row_idxs_lower, row_idxs_upper))
                self.jacob_col_idxs = np.concatenate((col_idxs_center, col_idxs_lower, col_idxs_upper))

            # "steady" convergence measures
            if solver.run_steady:
                self.d_sol_norm_l2 = 0.0
                self.d_sol_norm_l1 = 0.0
                self.d_sol_norm_history = np.zeros((solver.num_steps, 2), dtype=REAL_TYPE)

    def calc_d_sol_prim_d_sol_cons(self):
        """
        Compute the Jacobian of the conservative state w/r/t/ the primitive state

        This appears as Gamma^{-1} in the PERFORM documentation
        """

        # TODO: some repeated calculations
        # TODO: add option for preconditioning d_rho_d_press

        gas = self.gas_model

        gamma_matrix_inv = np.zeros((gas.num_eqs, gas.num_eqs, self.num_cells))

        # for clarity
        rho = self.sol_cons[0, :]
        press = self.sol_prim[0, :]
        vel = self.sol_prim[1, :]
        temp = self.sol_prim[2, :]
        mass_fracs = self.sol_prim[3:, :]

        d_rho_d_press = self.d_rho_d_press
        d_rho_d_temp = self.d_rho_d_temp
        d_rho_d_mass_frac = self.d_rho_d_mass_frac
        d_enth_d_press = self.d_enth_d_press
        d_enth_d_temp = self.d_enth_d_temp
        d_enth_d_mass_frac = self.d_enth_d_mass_frac
        h0 = self.h0

        # some reused terms
        d = rho * d_rho_d_press * d_enth_d_temp - d_rho_d_temp * (rho * d_enth_d_press - 1.0)
        vel_sq = np.square(vel)

        # density row
        gamma_matrix_inv[0, 0, :] = (
            rho * d_enth_d_temp
            + d_rho_d_temp * (h0 - vel_sq)
            + np.sum(
                mass_fracs * (d_rho_d_mass_frac * d_enth_d_temp[None, :] - d_rho_d_temp[None, :] * d_enth_d_mass_frac),
                axis=0,
            )
        ) / d
        gamma_matrix_inv[0, 1, :] = vel * d_rho_d_temp / d
        gamma_matrix_inv[0, 2, :] = -d_rho_d_temp / d
        gamma_matrix_inv[0, 3:, :] = (
            d_rho_d_temp[None, :] * d_enth_d_mass_frac - d_rho_d_mass_frac * d_enth_d_temp[None, :]
        ) / d[None, :]

        # momentum row
        gamma_matrix_inv[1, 0, :] = -vel / rho
        gamma_matrix_inv[1, 1, :] = 1.0 / rho

        # energy row
        gamma_matrix_inv[2, 0, :] = (
            -d_rho_d_press * (h0 - vel_sq)
            - (rho * d_enth_d_press - 1.0)
            + np.sum(
                mass_fracs
                * (
                    (rho * d_rho_d_press)[None, :] * d_enth_d_mass_frac
                    + d_rho_d_mass_frac * (rho * d_enth_d_press - 1.0)[None, :]
                ),
                axis=0,
            )
            / rho
        ) / d
        gamma_matrix_inv[2, 1, :] = -vel * d_rho_d_press / d
        gamma_matrix_inv[2, 2, :] = d_rho_d_press / d
        gamma_matrix_inv[2, 3:, :] = (
            -(
                (rho * d_rho_d_press)[None, :] * d_enth_d_mass_frac
                + d_rho_d_mass_frac * (rho * d_enth_d_press - 1.0)[None, :]
            )
            / (rho * d)[None, :]
        )

        # species row(s)
        gamma_matrix_inv[3:, 0, :] = -mass_fracs / rho[None, :]
        for i in range(3, gas.num_eqs):
            gamma_matrix_inv[i, i, :] = 1.0 / rho

        return gamma_matrix_inv

    def calc_d_sol_cons_d_sol_prim(self):
        """
        Compute the Jacobian of conservative state w/r/t the primitive state

        This appears as Gamma in the PERFORM documentation
        """

        # TODO: add option for preconditioning d_rho_d_press

        gas = self.gas_model

        gamma_matrix = np.zeros((gas.num_eqs, gas.num_eqs, self.num_cells))

        # for clarity
        rho = self.sol_cons[0, :]
        press = self.sol_prim[0, :]
        vel = self.sol_prim[1, :]
        temp = self.sol_prim[2, :]
        mass_fracs = self.sol_prim[3:, :]

        d_rho_d_press = self.d_rho_d_press
        d_rho_d_temp = self.d_rho_d_temp
        d_rho_d_mass_frac = self.d_rho_d_mass_frac
        d_enth_d_press = self.d_enth_d_press
        d_enth_d_temp = self.d_enth_d_temp
        d_enth_d_mass_frac = self.d_enth_d_mass_frac
        h0 = self.h0

        # density row
        gamma_matrix[0, 0, :] = d_rho_d_press
        gamma_matrix[0, 2, :] = d_rho_d_temp
        gamma_matrix[0, 3:, :] = d_rho_d_mass_frac

        # momentum row
        gamma_matrix[1, 0, :] = vel * d_rho_d_press
        gamma_matrix[1, 1, :] = rho
        gamma_matrix[1, 2, :] = vel * d_rho_d_temp
        gamma_matrix[1, 3:, :] = vel[None, :] * d_rho_d_mass_frac

        # total energy row
        gamma_matrix[2, 0, :] = d_rho_d_press * h0 + rho * d_enth_d_press - 1.0
        gamma_matrix[2, 1, :] = rho * vel
        gamma_matrix[2, 2, :] = d_rho_d_temp * h0 + rho * d_enth_d_temp
        gamma_matrix[2, 3:, :] = h0[None, :] * d_rho_d_mass_frac + rho[None, :] * d_enth_d_mass_frac

        # species row
        gamma_matrix[3:, 0, :] = mass_fracs[gas.mass_frac_slice, :] * d_rho_d_press[None, :]
        gamma_matrix[3:, 2, :] = mass_fracs[gas.mass_frac_slice, :] * d_rho_d_temp[None, :]
        for i in range(3, gas.num_eqs):
            for j in range(3, gas.num_eqs):
                gamma_matrix[i, j, :] = (i == j) * rho + mass_fracs[i - 3, :] * d_rho_d_mass_frac[j - 3, :]

        return gamma_matrix

    def res_jacob_assemble(self, center_block, lower_block, upper_block):
        """
        Reassemble residual Jacobian into a sparse 2D array for linear solve
        """

        # TODO: my God, this is still the single most expensive operation
        # How can this be any simpler/faster???
        # Preallocating "data" is *slower*

        data = np.concatenate((center_block.ravel("C"), lower_block.ravel("C"), upper_block.ravel("C")))
        res_jacob = csr_matrix(
            (data, (self.jacob_row_idxs, self.jacob_col_idxs)), shape=(self.jacob_dim, self.jacob_dim), dtype=REAL_TYPE
        )

        return res_jacob

    def calc_adaptive_dtau(self, mesh):
        """
        Adapt dtau for each cell based on user input constraints and local wave speed
        """

        gas_model = self.gas_model

        # compute initial dtau from input cfl and srf
        dtaum = 1.0 * mesh.dx / self.srf
        dtau = self.time_integrator.cfl * dtaum

        # limit by von Neumann number
        if self.visc_flux_name != "invisc":
            # TODO: calculating this is stupidly expensive
            self.dyn_visc_mix = gas_model.calc_mix_dynamic_visc(
                temperature=self.sol_prim[2, :], mass_fracs=self.sol_prim[3:, :]
            )
            nu = self.dyn_visc_mix / self.sol_cons[0, :]
            dtau = np.minimum(dtau, self.time_integrator.vnn * np.square(mesh.dx) / nu)
            dtaum = np.minimum(dtaum, 3.0 / nu)

        # limit dtau
        # TODO: implement solutionChangeLimitedTimeStep from gems_precon.f90

        return 1.0 / dtau

    def update_sol_hist(self):
        """
        Update time history of solution and RHS function for multi-stage time integrators
        """

        # primitive and conservative state history
        self.sol_hist_cons[1:] = self.sol_hist_cons[:-1]
        self.sol_hist_prim[1:] = self.sol_hist_prim[:-1]
        self.sol_hist_cons[0] = self.sol_cons.copy()
        self.sol_hist_prim[0] = self.sol_prim.copy()

        # TODO: RHS update should occur at the FIRST subiteration
        # 	right after the RHS is calculated
        # RHS function history
        self.rhs_hist[1:] = self.rhs_hist[:-1]
        self.rhs_hist[0] = self.rhs.copy()

    def update_snapshots(self, solver):

        store_idx = int((solver.iter - 1) / solver.out_interval) + 1

        if solver.prim_out:
            self.prim_snap[:, :, store_idx] = self.sol_prim
        if solver.cons_out:
            self.cons_snap[:, :, store_idx] = self.sol_cons
        if solver.source_out:
            self.source_snap[:, :, store_idx - 1] = self.source
        if solver.rhs_out:
            self.rhs_snap[:, :, store_idx - 1] = self.rhs

    def write_snapshots(self, solver, failed):
        """
        Save snapshot matrices to disk
        """

        unsteady_output_dir = solver.unsteady_output_dir

        # account for failed simulation dump
        # TODO: need to account for non-unity out_interval
        if failed:
            offset = 1
        else:
            offset = 2
        final_idx = int((solver.iter - 1) / solver.out_interval) + offset

        if solver.prim_out:
            sol_prim_file = os.path.join(unsteady_output_dir, "sol_prim_" + solver.sim_type + ".npy")
            np.save(sol_prim_file, self.prim_snap[:, :, :final_idx])

        if solver.cons_out:
            sol_cons_file = os.path.join(unsteady_output_dir, "sol_cons_" + solver.sim_type + ".npy")
            np.save(sol_cons_file, self.cons_snap[:, :, :final_idx])

        if solver.source_out:
            source_file = os.path.join(unsteady_output_dir, "source_" + solver.sim_type + ".npy")
            np.save(source_file, self.source_snap[:, :, : final_idx - 1])

        if solver.rhs_out:
            sol_rhs_file = os.path.join(unsteady_output_dir, "rhs_" + solver.sim_type + ".npy")
            np.save(sol_rhs_file, self.rhs_snap[:, :, : final_idx - 1])

    def write_restart_file(self, solver):
        """
        Write restart files containing primitive and conservative fields,
        and associated physical time
        """

        # TODO: write previous time step(s) for multi-step methods
        # 	to preserve time accuracy at restart

        # write restart file to zipped file
        restart_file = os.path.join(solver.restart_output_dir, "restart_file_" + str(solver.restart_iter) + ".npz")
        np.savez(restart_file, sol_time=solver.sol_time, sol_prim=self.sol_prim, sol_cons=self.sol_cons)

        # write iteration number files
        restartIterFile = os.path.join(solver.restart_output_dir, "restart_iter.dat")
        with open(restartIterFile, "w") as f:
            f.write(str(solver.restart_iter) + "\n")

        restart_phys_iter_file = os.path.join(
            solver.restart_output_dir, "restart_iter_" + str(solver.restart_iter) + ".dat"
        )
        with open(restart_phys_iter_file, "w") as f:
            f.write(str(solver.iter) + "\n")

        # iterate file count
        if solver.restart_iter < solver.num_restarts:
            solver.restart_iter += 1
        else:
            solver.restart_iter = 1

    def write_steady_data(self, solver):

        # write norm data to ASCII file
        steady_file = os.path.join(solver.unsteady_output_dir, "steady_convergence.dat")
        if solver.iter == 1:
            f = open(steady_file, "w")
        else:
            f = open(steady_file, "a")
        out_string = ("%8i %18.14f %18.14f\n") % (solver.time_iter - 1, self.d_sol_norm_l2, self.d_sol_norm_l1)
        f.write(out_string)
        f.close()

        # write field data
        sol_prim_file = os.path.join(solver.unsteady_output_dir, "sol_prim_steady.npy")
        np.save(sol_prim_file, self.sol_prim)

        sol_cons_file = os.path.join(solver.unsteady_output_dir, "sol_cons_steady.npy")
        np.save(sol_cons_file, self.sol_cons)

    def calc_d_sol_norms(self, solver, time_type):
        """
        Calculate and print solution change norms
        Note that output is ORDER OF MAGNITUDE of residual norm
        (i.e. 1e-X, where X is the order of magnitude)
        """

        if time_type == "implicit":
            d_sol = self.sol_hist_prim[0] - self.sol_hist_prim[1]
        else:
            # TODO: only valid for single-stage explicit schemes
            d_sol = self.sol_prim - self.sol_hist_prim[0]

        norm_l2, norm_l1 = self.calc_norms(d_sol, solver.res_norm_prim)

        norm_out_l2 = np.log10(norm_l2)
        norm_out_l1 = np.log10(norm_l1)
        out_string = ("%8i:   L2: %18.14f,   L1: %18.14f") % (solver.time_iter, norm_out_l2, norm_out_l1)
        print(out_string)

        self.d_sol_norm_l2 = norm_l2
        self.d_sol_norm_l1 = norm_l1
        self.d_sol_norm_history[solver.iter - 1, :] = [norm_l2, norm_l1]

    def calc_res_norms(self, solver, subiter):
        """
        Calculate and print linear solve residual norms

        Note that output is ORDER OF MAGNITUDE of residual norm (i.e. 1e-X, where X is the order of magnitude)
        """

        # TODO: pass conservative norm factors if running cons implicit solve
        norm_l2, norm_l1 = self.calc_norms(self.res, solver.res_norm_prim)

        # don't print for "steady" solve
        if not solver.run_steady:
            norm_out_l2 = np.log10(norm_l2)
            norm_out_l1 = np.log10(norm_l1)
            out_string = (str(subiter + 1) + ":\tL2: %18.14f, \tL1: %18.14f") % (norm_out_l2, norm_out_l1)
            print(out_string)

        self.res_norm_l2 = norm_l2
        self.res_norm_l1 = norm_l1
        self.res_norm_history[solver.iter - 1, :] = [norm_l2, norm_l1]

    def calc_norms(self, arr_in, norm_facs):
        """
        Compute L1 and L2 norms of arr_in

        arr_in assumed to be in [numVars, num_cells] order
        """

        arr_abs = np.abs(arr_in)

        # L2 norm
        arr_norm_l2 = np.sum(np.square(arr_abs), axis=1)
        arr_norm_l2[:] /= arr_in.shape[1]
        arr_norm_l2 /= np.square(norm_facs)
        arr_norm_l2 = np.sqrt(arr_norm_l2)
        arr_norm_l2 = np.mean(arr_norm_l2)

        # L1 norm
        arr_norm_l1 = np.sum(arr_abs, axis=1)
        arr_norm_l1[:] /= arr_in.shape[1]
        arr_norm_l1 /= norm_facs
        arr_norm_l1 = np.mean(arr_norm_l1)

        return arr_norm_l2, arr_norm_l1

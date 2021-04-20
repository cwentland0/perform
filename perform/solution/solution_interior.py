import os

import numpy as np
from scipy.sparse import csr_matrix

from perform.constants import REAL_TYPE, RES_NORM_PRIM_DEFAULT
from perform.solution.solution_phys import SolutionPhys


class SolutionInterior(SolutionPhys):
    """Physical solution of interior cells.

    This SolutionPhys represents the interior finite volume cells of a SolutionDomain,
    i.e. all cells except the boundary ghost cells. There is only one SolutionInterior per SolutionDomain.

    A few constructs, such as the source term, residual, residual Jacobian, etc., are only meaningful for the interior
    cells and so are represented here specifically.

    Additionally, this class provides member methods which handle the output of snapshot matrices and residual norms,
    as there are also only meaningful for the interior cells.

    Args:
        gas: GasModel associated with the SolutionDomain with which this SolutionPhys is associated.
        sol_prim_in: NumPy array of the primitive state profiles that this SolutionPhys represents.
        solver: SystemSolver containing global simulation parameters.
        num_cells: Number of finite volume cells represented by this SolutionPhys.
        num_reactions: Number of reactions to be modeled.
        time_int: TimeIntegrator associated with the SolutionDomain with which this SolutionInterior is associated.

    Attributes:
        wf:
            NumPy array of the rate-of-progress profiles for the num_reactions reactions,
            if modeling reactions by a finite-rate reaction model.
        source: NumPy array of the reaction source term profiles for the num_species species transport equations.
        rhs: NumPy array of the evaluation of the right-hand side function of the semi-discrete governing ODE.
        sol_hist_cons: List of NumPy arrays of the prior time_int.time_order conservative state profiles.
        sol_hist_prim: List of NumPy arrays of the prior time_int.time_order primitive state profiles.
        rhs_hist: List of NumPy arrays of the prior time_int.time_order RHS function profiles.
        prim_snap: NumPy array of the primitive state snapshot array to be written to disk.
        cons_snap: NumPy array of the conservative state snapshot array to be written to disk.
        source_snap: NumPy array of the source term snapshot array to be written to disk.
        rhs_snap: NumPy array of the RHS function snapshot array to be written to disk.
        res: NumPy array of the full-discrete residual function profile.
        res_norm_l2: L2 norm of the Newton iteration linear solve residual, normalized.
        res_norm_l1: L1 norm of the Newton iteration linear solve residual, normalized.
        res_norm_hist: NumPy array of the time history of the L2 and L1 linear solver residual norm.
        jacob_dim_first: Leading dimension of the residual Jacobian.
        jacob_dim_second: Trailing dimension of the residual Jacobian.
        jacob_row_idxs:
            NumPy array of row indices within the 2D residual Jacobian at which the
            flattened 3D residual Jacobian will be emplaced.
        jacob_col_idxs:
            NumPy array of column indices within the 2D residual Jacobian at which the
            flattened 3D residual Jacobian will be emplaced.
        d_sol_norm_l2: L2 norm of the primitive solution change, normalized.
        d_sol_norm_l1: L1 norm of the primitive solution change, normalized.
        d_sol_norm_hist: NumPy array of the time history of the L2 and L1 primitive solution change norm.
    """

    def __init__(self, gas, sol_prim_in, solver, num_cells, num_reactions, time_int):

        super().__init__(gas, num_cells, sol_prim_in=sol_prim_in)

        gas = self.gas_model

        if num_reactions > 0:
            self.wf = np.zeros((num_reactions, num_cells), dtype=REAL_TYPE)
        self.source = np.zeros((gas.num_species, num_cells), dtype=REAL_TYPE)
        self.rhs = np.zeros((gas.num_eqs, num_cells), dtype=REAL_TYPE)

        # Add bulk velocity and update state if requested
        if solver.vel_add != 0.0:
            self.sol_prim[1, :] += solver.vel_add
            self.update_state(from_cons=False)

        # Initializing time history
        self.sol_hist_cons = [self.sol_cons.copy()] * (time_int.time_order + 1)
        self.sol_hist_prim = [self.sol_prim.copy()] * (time_int.time_order + 1)

        # RHS storage for multi-stage schemes
        self.rhs_hist = [self.rhs.copy()] * (time_int.time_order + 1)

        # Snapshot storage matrices, store initial condition
        if solver.prim_out:
            self.prim_snap = np.zeros((gas.num_eqs, num_cells, solver.num_snaps + 1), dtype=REAL_TYPE)
            self.prim_snap[:, :, 0] = self.sol_prim.copy()
        if solver.cons_out:
            self.cons_snap = np.zeros((gas.num_eqs, num_cells, solver.num_snaps + 1), dtype=REAL_TYPE)
            self.cons_snap[:, :, 0] = self.sol_cons.copy()

        # These don't include the source/RHS associated with the final solution
        if solver.source_out:
            self.source_snap = np.zeros((gas.num_species, num_cells, solver.num_snaps), dtype=REAL_TYPE)
        if solver.rhs_out:
            self.rhs_snap = np.zeros((gas.num_eqs, num_cells, solver.num_snaps), dtype=REAL_TYPE)

        if (time_int.time_type == "implicit") or (solver.run_steady):
            # Norm normalization constants
            if (len(solver.res_norm_prim) == 1) and (solver.res_norm_prim[0] is None):
                solver.res_norm_prim = [None] * gas.num_eqs
            else:
                assert len(solver.res_norm_prim) == gas.num_eqs
            for var_idx in range(gas.num_eqs):
                if solver.res_norm_prim[var_idx] is None:
                    # 0: pressure, 1: velocity, 2: temperature, >=3: species
                    solver.res_norm_prim[var_idx] = RES_NORM_PRIM_DEFAULT[min(var_idx, 3)]

            # Residual norm storage
            if time_int.time_type == "implicit":

                self.res = np.zeros((gas.num_eqs, num_cells), dtype=REAL_TYPE)
                self.res_norm_l2 = 0.0
                self.res_norm_l1 = 0.0
                self.res_norm_hist = np.zeros((solver.num_steps, 2), dtype=REAL_TYPE)

                if (time_int.dual_time) and (time_int.adapt_dtau):
                    self.srf = np.zeros(num_cells, dtype=REAL_TYPE)

                # CSR matrix indices
                num_elements = gas.num_eqs ** 2 * num_cells
                self.jacob_dim_first = gas.num_eqs * num_cells
                self.jacob_dim_second = self.jacob_dim_first

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

            # "Steady" convergence measures
            if solver.run_steady:
                self.d_sol_norm_l2 = 0.0
                self.d_sol_norm_l1 = 0.0
                self.d_sol_norm_hist = np.zeros((solver.num_steps, 2), dtype=REAL_TYPE)

    def calc_d_sol_prim_d_sol_cons(self, samp_idxs=np.s_[:]):
        """Compute the Jacobian of the primitive state w/r/t/ the conservative state

        This Jacobian is calculated when computing the approximate residual Jacobian w/r/t the conservative
        state when dual_time == False. It is also implicitly used when calculating the Roe dissipation term,
        but is not explicitly calculated there.

        Assumes that the stagnation enthalpy, derivatives of density,
        and derivatives of stagnation enthalpy have already been computed.

        This appears as Gamma^{-1} in the solver theory documentation, please refer to the solver theory documentation
        for a detailed derivation of this matrix.

        Args:
            samp_idxs:
                Either a NumPy slice or NumPy array for selecting sampled cells to compute the Jacobian at.
                Used for hyper-reduction of projection-based reduced-order models.

        Returns:
            3D NumPy array of the Jacobian of the primitive state w/r/t the conservative state.
        """

        # TODO: some repeated calculations
        # TODO: add option for preconditioning d_rho_d_press

        gas = self.gas_model

        # Initialize Jacobian
        if type(samp_idxs) is slice:
            num_cells = self.num_cells
        else:
            num_cells = samp_idxs.shape[0]
        gamma_matrix_inv = np.zeros((gas.num_eqs, gas.num_eqs, num_cells))

        # For clarity
        rho = self.sol_cons[0, samp_idxs]
        vel = self.sol_prim[1, samp_idxs]
        mass_fracs = self.sol_prim[3:, samp_idxs]

        d_rho_d_press = self.d_rho_d_press[samp_idxs]
        d_rho_d_temp = self.d_rho_d_temp[samp_idxs]
        d_rho_d_mass_frac = self.d_rho_d_mass_frac[:, samp_idxs]
        d_enth_d_press = self.d_enth_d_press[samp_idxs]
        d_enth_d_temp = self.d_enth_d_temp[samp_idxs]
        d_enth_d_mass_frac = self.d_enth_d_mass_frac[:, samp_idxs]
        h0 = self.h0[samp_idxs]

        # Some reused terms
        d = rho * d_rho_d_press * d_enth_d_temp - d_rho_d_temp * (rho * d_enth_d_press - 1.0)
        vel_sq = np.square(vel)

        # Density row
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

        # Momentum row
        gamma_matrix_inv[1, 0, :] = -vel / rho
        gamma_matrix_inv[1, 1, :] = 1.0 / rho

        # Energy row
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

        # Species row(s)
        gamma_matrix_inv[3:, 0, :] = -mass_fracs / rho[None, :]
        for i in range(3, gas.num_eqs):
            gamma_matrix_inv[i, i, :] = 1.0 / rho

        return gamma_matrix_inv

    def calc_d_sol_cons_d_sol_prim(self, samp_idxs=np.s_[:]):
        """Compute the Jacobian of the conservative state w/r/t/ the primitive state

        This Jacobian is calculated when computing the residual Jacobian w/r/t the primitive
        state when dual_time == True.

        Assumes that the stagnation enthalpy, derivatives of density,
        and derivatives of stagnation enthalpy have already been computed.

        This appears as Gamma in the solver theory documentation, please refer to
        the solver theory documentation for a detailed derivation of this matrix.

        Args:
            samp_idxs:
                Either a NumPy slice or NumPy array for selecting sampled cells to compute the Jacobian at.
                Used for hyper-reduction of projection-based reduced-order models.

        Returns:
            3D NumPy array of the Jacobian of the conservative state w/r/t the primitive state.
        """

        # TODO: add option for preconditioning d_rho_d_press

        gas = self.gas_model

        # Initialize Jacobian
        if type(samp_idxs) is slice:
            num_cells = self.num_cells
        else:
            num_cells = samp_idxs.shape[0]
        gamma_matrix = np.zeros((gas.num_eqs, gas.num_eqs, num_cells))

        # For clarity
        rho = self.sol_cons[0, samp_idxs]
        vel = self.sol_prim[1, samp_idxs]
        mass_fracs = self.sol_prim[3:, samp_idxs]

        d_rho_d_press = self.d_rho_d_press[samp_idxs]
        d_rho_d_temp = self.d_rho_d_temp[samp_idxs]
        d_rho_d_mass_frac = self.d_rho_d_mass_frac[:, samp_idxs]
        d_enth_d_press = self.d_enth_d_press[samp_idxs]
        d_enth_d_temp = self.d_enth_d_temp[samp_idxs]
        d_enth_d_mass_frac = self.d_enth_d_mass_frac[:, samp_idxs]
        h0 = self.h0[samp_idxs]

        # Density row
        gamma_matrix[0, 0, :] = d_rho_d_press
        gamma_matrix[0, 2, :] = d_rho_d_temp
        gamma_matrix[0, 3:, :] = d_rho_d_mass_frac

        # Momentum row
        gamma_matrix[1, 0, :] = vel * d_rho_d_press
        gamma_matrix[1, 1, :] = rho
        gamma_matrix[1, 2, :] = vel * d_rho_d_temp
        gamma_matrix[1, 3:, :] = vel[None, :] * d_rho_d_mass_frac

        # Total energy row
        gamma_matrix[2, 0, :] = d_rho_d_press * h0 + rho * d_enth_d_press - 1.0
        gamma_matrix[2, 1, :] = rho * vel
        gamma_matrix[2, 2, :] = d_rho_d_temp * h0 + rho * d_enth_d_temp
        gamma_matrix[2, 3:, :] = h0[None, :] * d_rho_d_mass_frac + rho[None, :] * d_enth_d_mass_frac

        # Species row
        gamma_matrix[3:, 0, :] = mass_fracs[gas.mass_frac_slice, :] * d_rho_d_press[None, :]
        gamma_matrix[3:, 2, :] = mass_fracs[gas.mass_frac_slice, :] * d_rho_d_temp[None, :]
        for i in range(3, gas.num_eqs):
            for j in range(3, gas.num_eqs):
                gamma_matrix[i, j, :] = (i == j) * rho + mass_fracs[i - 3, :] * d_rho_d_mass_frac[j - 3, :]

        return gamma_matrix

    def res_jacob_assemble(self, center_block, lower_block, upper_block):
        """Assembles residual Jacobian into a sparse 2D matrix for Newton's method linear solve.

        The components of the residual Jacobian are provided as 3D arrays representing the center, lower,
        and upper block diagonal of the residual Jacobian if the residual were flattened in column-major
        order (i.e. variables first, then cells). This makes populating the components easier, but makes
        calculating the linear solve more tedious. Thus, the residual Jacobian is constructed for a residual
        flattened in row-major order (i.e. cells first, then variables).

        Note that the returned Jacobian is a scipy.sparse matrix; care must be taken when performing math operations
        with this as the resulting object may devolve to a NumPy matrix, which is deprecated. If this problem
        cannot be avoided, use .toarray() to convert the sparse matrix to a NumPy array. This is not ideal as it
        generates a large dense matrix.

        Args:
            center_block: 3D NumPy array of the center block diagonal of column-major residual Jacobian.
            lower_block: 3D NumPy array of the lower block diagonal of column-major residual Jacobian.
            upper_block: 3D NumPy array of the upper block diagonal of column-major residual Jacobian.

        Returns:
            scipy.sparse.csr_matrix of row-major residual Jacobian.
        """

        # TODO: my God, this is still the single most expensive operation
        # How can this be any simpler/faster???
        # Preallocating "jacob_diags" is *slower*

        jacob_diags = np.concatenate((center_block.ravel("C"), lower_block.ravel("C"), upper_block.ravel("C")))
        res_jacob = csr_matrix(
            (jacob_diags, (self.jacob_row_idxs, self.jacob_col_idxs)),
            shape=(self.jacob_dim_first, self.jacob_dim_second),
            dtype=REAL_TYPE,
        )

        return res_jacob

    def calc_adaptive_dtau(self, mesh):
        """Adapt dtau for each cell based on user input constraints and local wave speed.

        This function is intended to improve dual time-stepping robustness, but mostly acts to slow convergence.
        For now, I recommend not setting adapt_dtau until this is completed.

        Args:
            mesh: Mesh associated with SolutionDomain with which this SolutionPhys is associated.

        Returns:
            Reciprocal of adapted dtau profile.
        """

        gas_model = self.gas_model

        # Compute initial dtau from input cfl and srf
        dtaum = 1.0 * mesh.dx / self.srf
        dtau = self.time_integrator.cfl * dtaum

        # Limit by von Neumann number
        if self.visc_flux_name != "invisc":
            # TODO: calculating this is stupidly expensive
            self.dyn_visc_mix = gas_model.calc_mix_dynamic_visc(
                temperature=self.sol_prim[2, :], mass_fracs=self.sol_prim[3:, :]
            )
            nu = self.dyn_visc_mix / self.sol_cons[0, :]
            dtau = np.minimum(dtau, self.time_integrator.vnn * np.square(mesh.dx) / nu)
            dtaum = np.minimum(dtaum, 3.0 / nu)

        # Limit dtau
        # TODO: finish implementation

        return 1.0 / dtau

    def update_sol_hist(self):
        """Update time history of solution and RHS function for multi-stage time integrators.

        After each physical time iteration, the primitive solution, conservative solution, and RHS profile
        histories are pushed back by one and the first entry is replaced by the current time step's profiles.
        """

        # Primitive and conservative state history
        self.sol_hist_cons[1:] = self.sol_hist_cons[:-1]
        self.sol_hist_prim[1:] = self.sol_hist_prim[:-1]
        self.sol_hist_cons[0] = self.sol_cons.copy()
        self.sol_hist_prim[0] = self.sol_prim.copy()

        # RHS function history
        self.rhs_hist[1:] = self.rhs_hist[:-1]
        self.rhs_hist[0] = self.rhs.copy()

    def update_snapshots(self, solver):
        """Update snapshot arrays.

        Adds current solution, source, RHS, etc. profiles to snapshots arrays.
        At the end of a simulation (completed or failed), these will be written to disk.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

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
        """Save snapshot matrices to disk after completed/failed simulation.

        Args:
            solver: SystemSolver containing global simulation parameters.
            failed: Boolean flag indicating whether a simulation has failed before completion.
        """

        unsteady_output_dir = solver.unsteady_output_dir

        # Account for failed simulation dump
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
        """Write restart files to disk.

        Restart files contain the primitive and conservative fields current associated physical time.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        # Write restart file to zipped file
        restart_file = os.path.join(solver.restart_output_dir, "restart_file_" + str(solver.restart_iter) + ".npz")
        np.savez(restart_file, sol_time=solver.sol_time, sol_prim=self.sol_prim, sol_cons=self.sol_cons)

        # Write iteration number files
        restartIterFile = os.path.join(solver.restart_output_dir, "restart_iter.dat")
        with open(restartIterFile, "w") as f:
            f.write(str(solver.restart_iter) + "\n")

        restart_phys_iter_file = os.path.join(
            solver.restart_output_dir, "restart_iter_" + str(solver.restart_iter) + ".dat"
        )
        with open(restart_phys_iter_file, "w") as f:
            f.write(str(solver.iter) + "\n")

        # Iterate file count
        if solver.restart_iter < solver.num_restarts:
            solver.restart_iter += 1
        else:
            solver.restart_iter = 1

    def write_steady_data(self, solver):
        """Write "steady" solve "convergence" norms and current solution profiles to disk.

        The "convergence" norms file provides a history of the steady solve convergence at each iteration,
        while the primitive and conservative solution file are continuously overwritten with the most recent solution.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        # Write norm data to ASCII file
        steady_file = os.path.join(solver.unsteady_output_dir, "steady_convergence.dat")
        if solver.iter == 1:
            f = open(steady_file, "w")
        else:
            f = open(steady_file, "a")
        out_string = ("%8i %18.14f %18.14f\n") % (solver.time_iter - 1, self.d_sol_norm_l2, self.d_sol_norm_l1)
        f.write(out_string)
        f.close()

        # Write field data
        sol_prim_file = os.path.join(solver.unsteady_output_dir, "sol_prim_steady.npy")
        np.save(sol_prim_file, self.sol_prim)

        sol_cons_file = os.path.join(solver.unsteady_output_dir, "sol_cons_steady.npy")
        np.save(sol_cons_file, self.sol_cons)

    def calc_norms(self, arr_in, norm_facs):
        """Compute average, normalized L1 and L2 norms of an array.

        In reality, this computes the RMS measure as it is divided by the square root of the number of elements.
        arr_in assumed to be in [num_vars, num_cells] order, as is the case for, e.g., sol_prim and sol_cons.

        Args:
            arr_in: NumPy array for which norms are to be calculated.
            norm_facs: NumPy array of factors by which to normalize each profile in arr_in before computing norms.

        Returns:
            Average, normalized L2 and L1 norms of arr_in.
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

    def calc_d_sol_norms(self, solver, time_type):
        """Calculate and print solution change norms for "steady" solve "convergence".

        Computes L2 and L1 norms of the change in the primitive solution between time steps.
        This measure will be used to determine whether the "steady" solve has "converged".

        Note that outputs are orders of magnitude (i.e. 1e-X, where X is the order of magnitude)

        Args:
            solver: SystemSolver containing global simulation parameters.
            time_type: Either "explicit" or "implicit", affects how solution profiles are retrieved.
        """

        # Calculate solution change and its norms
        if time_type == "implicit":
            d_sol = self.sol_hist_prim[0] - self.sol_hist_prim[1]
        else:
            d_sol = self.sol_prim - self.sol_hist_prim[0]

        norm_l2, norm_l1 = self.calc_norms(d_sol, solver.res_norm_prim)

        # Print to terminal
        norm_out_l2 = np.log10(norm_l2)
        norm_out_l1 = np.log10(norm_l1)
        out_string = ("%8i:   L2: %18.14f,   L1: %18.14f") % (solver.time_iter, norm_out_l2, norm_out_l1)
        print(out_string)

        self.d_sol_norm_l2 = norm_l2
        self.d_sol_norm_l1 = norm_l1
        self.d_sol_norm_hist[solver.iter - 1, :] = [norm_l2, norm_l1]

    def calc_res_norms(self, solver, subiter):
        """Calculate and print implicit time integration linear solve residual norms.

        Computes L2 and L1 norms of the Newton's method iterative solve for implicit time integration.
        This measure will be used to determine whether Newton's method has converged.

        Note that outputs are orders of magnitude (i.e. 1e-X, where X is the order of magnitude)

        Args:
            solver: SystemSolver containing global simulation parameters.
            subiter: Time step subiteration number, for terminal output.
        """

        norm_l2, norm_l1 = self.calc_norms(self.res, solver.res_norm_prim)

        # Don't print for "steady" solve
        if not solver.run_steady:
            norm_out_l2 = np.log10(norm_l2)
            norm_out_l1 = np.log10(norm_l1)
            out_string = (str(subiter + 1) + ":\tL2: %18.14f, \tL1: %18.14f") % (norm_out_l2, norm_out_l1)
            print(out_string)

        self.res_norm_l2 = norm_l2
        self.res_norm_l1 = norm_l1
        self.res_norm_hist[solver.iter - 1, :] = [norm_l2, norm_l1]

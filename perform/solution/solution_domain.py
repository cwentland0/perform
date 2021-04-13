import os

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

from perform.constants import REAL_TYPE
from perform.input_funcs import get_initial_conditions, catch_list, catch_input, read_input_file
from perform.mesh import Mesh
from perform.solution.solution_phys import SolutionPhys
from perform.solution.solution_interior import SolutionInterior
from perform.solution.solution_boundary.solution_inlet import SolutionInlet
from perform.solution.solution_boundary.solution_outlet import SolutionOutlet
from perform.time_integrator import get_time_integrator

# flux schemes
# TODO: make an __init__.py with get_flux_scheme()
from perform.flux.invisc_flux.roe_invisc_flux import RoeInviscFlux
from perform.flux.visc_flux.standard_visc_flux import StandardViscFlux

# gradient limiters
# TODO: make an __init__.py with get_limiter()
from perform.limiter.barth_jesp_limiter import BarthJespLimiter
from perform.limiter.venkat_limiter import VenkatLimiter

# gas models
# TODO: make an __init__.py with get_gas_model()
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas

# reaction models
# TODO: make an __init__.py with get_reaction_model()
from perform.reaction_model.finite_rate_irrev_reaction import FiniteRateIrrevReaction


class SolutionDomain:
    """
    Container class for interior and boundary physical solutions
    """

    def __init__(self, solver):

        param_dict = solver.param_dict

        # spatial domain
        mesh_file = str(param_dict["mesh_file"])
        mesh_dict = read_input_file(mesh_file)
        self.mesh = Mesh(mesh_dict)

        # gas model
        chem_file = str(param_dict["chem_file"])
        gas_dict = read_input_file(chem_file)
        gas_model_name = catch_input(gas_dict, "gas_model", "cpg")
        if gas_model_name == "cpg":
            self.gas_model = CaloricallyPerfectGas(gas_dict)
        else:
            raise ValueError("Ivalid choice of gas_model: " + gas_model_name)
        gas = self.gas_model

        # reaction model
        reaction_model_name = catch_input(gas_dict, "reaction_model", "none")
        if reaction_model_name == "none":
            assert solver.source_off, "Must provide a valid reaction_model_name if source_off = False"
            num_reactions = 0
        else:
            if reaction_model_name == "fr_irrev":
                self.reaction_model = FiniteRateIrrevReaction(gas, gas_dict)
            else:
                raise ValueError("Invalid choice of reaction_model: " + reaction_model_name)

            num_reactions = self.reaction_model.num_reactions

        # time integrator
        self.time_integrator = get_time_integrator(solver.time_scheme, param_dict)

        # solution
        sol_prim_init = get_initial_conditions(self, solver)
        self.sol_int = SolutionInterior(
            gas, sol_prim_init, solver, self.mesh.num_cells, num_reactions, self.time_integrator
        )
        self.sol_inlet = SolutionInlet(gas, solver)
        self.sol_outlet = SolutionOutlet(gas, solver)

        # flux scheme
        self.invisc_flux_name = catch_input(param_dict, "invisc_flux_scheme", "roe")
        self.visc_flux_name = catch_input(param_dict, "visc_flux_scheme", "invisc")

        # inviscid flux scheme
        if self.invisc_flux_name == "roe":
            self.invisc_flux_scheme = RoeInviscFlux(self)
            # TODO: move this to the actual flux class
            ones_prof = np.ones((self.gas_model.num_eqs, self.sol_int.num_cells + 1), dtype=REAL_TYPE)
            self.sol_ave = SolutionPhys(gas, self.sol_int.num_cells + 1, sol_prim_in=ones_prof)
        else:
            raise ValueError("Invalid entry for invisc_flux_name: " + str(self.invisc_flux_name))

        # viscous flux scheme
        if self.visc_flux_name == "invisc":
            pass
        elif self.visc_flux_name == "standard":
            self.visc_flux_scheme = StandardViscFlux(self)
        else:
            raise ValueError("Invalid entry for visc_flux_name: " + str(self.visc_flux_name))

        # higher-order reconstructions and gradient limiters
        self.space_order = catch_input(param_dict, "space_order", 1)
        assert self.space_order >= 1, "space_order must be a positive integer"
        if self.space_order > 1:
            self.grad_limiter_name = catch_input(param_dict, "grad_limiter", "none")

            if self.grad_limiter_name == "none":
                pass
            elif self.grad_limiter_name == "barth":
                self.grad_limiter = BarthJespLimiter()
            elif self.grad_limiter_name == "venkat":
                self.grad_limiter = VenkatLimiter()
            else:
                raise ValueError("Invalid entry for grad_limiter_name: " + str(self.grad_limiter_name))

        # for flux calculations
        ones_prof = np.ones((self.gas_model.num_eqs, self.sol_int.num_cells + 1), dtype=REAL_TYPE)
        self.sol_left = SolutionPhys(gas, self.sol_int.num_cells + 1, sol_prim_in=ones_prof)
        self.sol_right = SolutionPhys(gas, self.sol_int.num_cells + 1, sol_prim_in=ones_prof)

        # to avoid repeated concatenation of ghost cell states
        self.sol_prim_full = np.zeros(
            (self.gas_model.num_eqs, (self.sol_inlet.num_cells + self.sol_int.num_cells + self.sol_outlet.num_cells)),
            dtype=REAL_TYPE,
        )
        self.sol_cons_full = np.zeros(self.sol_prim_full.shape, dtype=REAL_TYPE)

        # probe storage (as this can include boundaries as well)
        self.probe_locs = catch_list(param_dict, "probe_locs", [None])
        self.probe_vars = catch_list(param_dict, "probe_vars", [None])
        if (self.probe_locs[0] is not None) and (self.probe_vars[0] is not None):
            self.num_probes = len(self.probe_locs)
            self.num_probe_vars = len(self.probe_vars)
            self.probe_vals = np.zeros((self.num_probes, self.num_probe_vars, solver.num_steps), dtype=REAL_TYPE)

            # get probe locations
            self.probe_idxs = [None] * self.num_probes
            self.probe_secs = [None] * self.num_probes
            for idx, probe_loc in enumerate(self.probe_locs):
                if probe_loc > self.mesh.x_right:
                    self.probe_secs[idx] = "outlet"
                elif probe_loc < self.mesh.x_left:
                    self.probe_secs[idx] = "inlet"
                else:
                    self.probe_secs[idx] = "interior"
                    self.probe_idxs[idx] = np.abs(self.mesh.x_cell - probe_loc).argmin()

            assert not (
                (("outlet" in self.probe_secs) or ("inlet" in self.probe_secs))
                and (("source" in self.probe_vars) or ("rhs" in self.probe_vars))
            ), "Cannot probe source or rhs in inlet/outlet"

        else:
            self.num_probes = 0

        # copy this for use with plotting functions
        solver.num_probes = self.num_probes
        solver.probe_vars = self.probe_vars

        # TODO: include initial conditions in probe_vals, time_vals
        self.time_vals = np.linspace(
            solver.dt * (solver.time_iter),
            solver.dt * (solver.time_iter - 1 + solver.num_steps),
            solver.num_steps,
            dtype=REAL_TYPE,
        )

        # for compatability with hyper-reduction
        # are overwritten if actually using hyper-reduction
        self.num_samp_cells = self.mesh.num_cells
        self.num_flux_faces = self.mesh.num_cells + 1
        self.num_grad_cells = self.mesh.num_cells
        # indices of directly sampled cells, within sol_prim/cons
        self.direct_samp_idxs = np.arange(0, self.mesh.num_cells)
        # indices of "left" cells for flux calcs, within sol_prim/cons_full
        self.flux_samp_left_idxs = np.s_[:-1]
        # indices of "right" cells for flux calcs, within sol_prim/cons_full
        self.flux_samp_right_idxs = np.s_[1:]
        # indices of cells for which gradients need to be calculated, within sol_prim/cons_full
        self.grad_idxs = np.arange(1, self.mesh.num_cells + 1)
        # indices of gradient cells and their immediate neighbors, within sol_prim/cons_full
        self.grad_neigh_idxs = np.s_[:]
        # indices within gradient neighbor indices to extract gradient cells, excluding boundaries
        self.grad_neigh_extract = np.s_[1:-1]
        # indices within flux_samp_left_idxs which map to indices of grad_idxs
        self.flux_left_extract = np.s_[1:]
        # indices within flux_samp_right_idxs which map to indices of grad_idxs
        self.flux_right_extract = np.s_[:-1]
        # indices within grad_idxs which map to indices of flux_samp_left_idxs
        self.grad_left_extract = np.s_[:]
        # indices within grad_idxs which map to indices of flux_samp_right_idxs
        self.grad_right_extract = np.s_[:]
        # indices of flux array which correspond to left face of cell and map to direct_samp_idxs
        self.flux_rhs_idxs = np.arange(0, self.mesh.num_cells)

        # for computing Jacobians
        self.jacob_left_samp = self.flux_rhs_idxs[1:].copy()
        self.jacob_right_samp = self.flux_rhs_idxs[:-1].copy() + 1

        # indices for computing Gamma inverse
        # TODO: remove these once conservative Jacobians are implemented
        self.gamma_idxs = np.s_[:]
        self.gamma_idxs_center = np.s_[:]
        self.gamma_idxs_left = np.s_[:-1]
        self.gamma_idxs_right = np.s_[1:]

    def fill_sol_full(self):
        """
        Fill sol_prim_full and sol_cons_full from interior and ghost cells
        """

        # TODO: hyper-reduction indices

        sol_int = self.sol_int
        sol_inlet = self.sol_inlet
        sol_outlet = self.sol_outlet

        idx_in = sol_inlet.num_cells
        idx_int = idx_in + sol_int.num_cells

        # sol_prim_full
        self.sol_prim_full[:, :idx_in] = sol_inlet.sol_prim.copy()
        self.sol_prim_full[:, idx_in:idx_int] = sol_int.sol_prim.copy()
        self.sol_prim_full[:, idx_int:] = sol_outlet.sol_prim.copy()

        # sol_cons_full
        self.sol_cons_full[:, :idx_in] = sol_inlet.sol_cons.copy()
        self.sol_cons_full[:, idx_in:idx_int] = sol_int.sol_cons.copy()
        self.sol_cons_full[:, idx_int:] = sol_outlet.sol_cons.copy()

    def advance_iter(self, solver):
        """
        Advance physical solution forward one time iteration
        """

        if not solver.run_steady:
            print("Iteration " + str(solver.iter))

        for self.time_integrator.subiter in range(self.time_integrator.subiter_max):

            self.advance_subiter(solver)

            # iterative solver convergence
            if self.time_integrator.time_type == "implicit":
                self.sol_int.calc_res_norms(solver, self.time_integrator.subiter)

                if self.sol_int.res_norm_l2 < self.time_integrator.res_tol:
                    break

        # "steady" convergence
        if solver.run_steady:
            self.sol_int.calc_d_sol_norms(solver, self.time_integrator.time_type)

        self.sol_int.update_sol_hist()

    def advance_subiter(self, solver):
        """
        Advance physical solution forward one subiteration of time integrator
        """

        self.calc_rhs(solver)

        sol_int = self.sol_int
        gas_model = self.gas_model
        mesh = self.mesh

        if self.time_integrator.time_type == "implicit":

            res = self.time_integrator.calc_residual(sol_int.sol_hist_cons, sol_int.rhs, solver)
            res_jacob = self.calc_res_jacob(solver)

            d_sol = spsolve(res_jacob, res.ravel("C"))

            # if solving in dual time, solving for primitive state
            if self.time_integrator.dual_time:
                sol_int.sol_prim += d_sol.reshape((gas_model.num_eqs, mesh.num_cells), order="C")

            else:
                sol_int.sol_cons += d_sol.reshape((gas_model.num_eqs, mesh.num_cells), order="C")

            sol_int.update_state(from_cons=(not self.time_integrator.dual_time))
            sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
            sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

            # use sol_int.res to store linear solve residual
            res = res_jacob @ d_sol - res.ravel("C")
            sol_int.res = np.reshape(res, (gas_model.num_eqs, mesh.num_cells), order="C")

        else:

            d_sol = self.time_integrator.solve_sol_change(sol_int.rhs)
            sol_int.sol_cons = sol_int.sol_hist_cons[0] + d_sol
            sol_int.update_state(from_cons=True)

    def calc_rhs(self, solver):
        """
        Compute rhs function
        """

        sol_int = self.sol_int
        sol_inlet = self.sol_inlet
        sol_outlet = self.sol_outlet
        sol_prim_full = self.sol_prim_full
        sol_cons_full = self.sol_cons_full
        samp_idxs = self.direct_samp_idxs
        gas = self.gas_model

        # compute ghost cell state (if adjacent cell is sampled)
        # TODO: update this after higher-order contribution?
        # TODO: adapt pass to calc_boundary_state() depending on space scheme
        if samp_idxs[0] == 0:
            sol_inlet.calc_boundary_state(
                solver.sol_time, self.space_order, sol_prim=sol_int.sol_prim[:, :2], sol_cons=sol_int.sol_cons[:, :2]
            )
        if samp_idxs[-1] == (self.mesh.num_cells - 1):
            sol_outlet.calc_boundary_state(
                solver.sol_time, self.space_order, sol_prim=sol_int.sol_prim[:, -2:], sol_cons=sol_int.sol_cons[:, -2:]
            )

        self.fill_sol_full()  # fill sol_prim_full and sol_cons_full

        # first-order approx at faces
        sol_left = self.sol_left
        sol_right = self.sol_right
        sol_left.sol_prim[:, :] = sol_prim_full[:, self.flux_samp_left_idxs]
        sol_left.sol_cons[:, :] = sol_cons_full[:, self.flux_samp_left_idxs]
        sol_right.sol_prim[:, :] = sol_prim_full[:, self.flux_samp_right_idxs]
        sol_right.sol_cons[:, :] = sol_cons_full[:, self.flux_samp_right_idxs]

        # add higher-order contribution
        # TODO: if not dual_time, reconstruct conservative variables instead of primitive
        if self.space_order > 1:
            sol_prim_grad = self.calc_cell_gradients()
            sol_left.sol_prim[:, self.flux_left_extract] += (self.mesh.dx / 2.0) * sol_prim_grad[
                :, self.grad_left_extract
            ]
            sol_right.sol_prim[:, self.flux_right_extract] -= (self.mesh.dx / 2.0) * sol_prim_grad[
                :, self.grad_right_extract
            ]
            sol_left.calc_state_from_prim()
            sol_right.calc_state_from_prim()

        # compute fluxes
        flux = self.invisc_flux_scheme.calc_flux(self)
        if self.visc_flux_name != "invisc":
            flux -= self.visc_flux_scheme.calc_flux(self)

        # compute RHS
        self.sol_int.rhs[:, samp_idxs] = flux[:, self.flux_rhs_idxs] - flux[:, self.flux_rhs_idxs + 1]
        sol_int.rhs[:, samp_idxs] /= self.mesh.dx

        # compute source term
        if not solver.source_off:
            source, wf = self.reaction_model.calc_source(self.sol_int, solver.dt, samp_idxs=samp_idxs)

            sol_int.source[gas.mass_frac_slice[:, None], samp_idxs[None, :]] = source
            sol_int.wf[:, samp_idxs] = wf

            sol_int.rhs[3:, samp_idxs] += sol_int.source[:, samp_idxs]

    def calc_cell_gradients(self):
        """
        Compute cell-centered gradients for higher-order face reconstructions

        Also calculate gradient limiters if requested
        """

        # TODO: for dual_time = False,
        # 	should be calculating conservative variable gradients

        # Compute gradients via finite difference stencil
        sol_prim_grad = np.zeros((self.gas_model.num_eqs, self.num_grad_cells), dtype=REAL_TYPE)
        if self.space_order == 2:
            sol_prim_grad = (0.5 / self.mesh.dx) * (
                self.sol_prim_full[:, self.grad_idxs + 1] - self.sol_prim_full[:, self.grad_idxs - 1]
            )
        else:
            raise ValueError("Order " + str(self.space_order) + " gradient calculations not implemented")

        # compute gradient limiter and limit gradient, if requested
        if self.grad_limiter_name != "none":
            phi = self.grad_limiter.calc_limiter(self, sol_prim_grad)
            sol_prim_grad = sol_prim_grad * phi

        return sol_prim_grad

    def calc_res_jacob(self, solver):
        """
        Compute Jacobian of residual
        """

        sol_int = self.sol_int
        gas = self.gas_model
        samp_idxs = self.gamma_idxs

        # enthalpies
        sol_int.hi[:, samp_idxs] = gas.calc_spec_enth(sol_int.sol_prim[2, samp_idxs])
        sol_int.h0[samp_idxs] = gas.calc_stag_enth(
            sol_int.sol_prim[1, samp_idxs], sol_int.mass_fracs_full[:, samp_idxs], spec_enth=sol_int.hi[:, samp_idxs],
        )

        # density derivatives
        (
            sol_int.d_rho_d_press[samp_idxs],
            sol_int.d_rho_d_temp[samp_idxs],
            sol_int.d_rho_d_mass_frac[:, samp_idxs],
        ) = gas.calc_dens_derivs(
            sol_int.sol_cons[0, samp_idxs],
            wrt_press=True,
            pressure=sol_int.sol_prim[0, samp_idxs],
            wrt_temp=True,
            temperature=sol_int.sol_prim[2, samp_idxs],
            wrt_spec=True,
            mix_mol_weight=sol_int.mw_mix[samp_idxs],
        )

        # stagnation enthalpy derivatives
        (
            sol_int.d_enth_d_press[samp_idxs],
            sol_int.d_enth_d_temp[samp_idxs],
            sol_int.d_enth_d_mass_frac[:, samp_idxs],
        ) = gas.calc_stag_enth_derivs(
            wrt_press=True,
            wrt_temp=True,
            mass_fracs=sol_int.sol_prim[3:, samp_idxs],
            wrt_spec=True,
            spec_enth=sol_int.hi[:, samp_idxs],
        )

        # Flux jacobians
        (d_flux_d_sol_prim, d_flux_d_sol_prim_left, d_flux_d_sol_prim_right) = self.invisc_flux_scheme.calc_jacob_prim(
            self
        )

        if self.visc_flux_name != "invisc":
            (
                d_visc_flux_d_sol_prim,
                d_visc_flux_d_sol_prim_left,
                d_visc_flux_d_sol_prim_right,
            ) = self.visc_flux_scheme.calc_jacob_prim(self)

            d_flux_d_sol_prim += d_visc_flux_d_sol_prim
            d_flux_d_sol_prim_left += d_visc_flux_d_sol_prim_left
            d_flux_d_sol_prim_right += d_visc_flux_d_sol_prim_right

        d_flux_d_sol_prim /= self.mesh.dx
        d_flux_d_sol_prim_left /= self.mesh.dx
        d_flux_d_sol_prim_right /= self.mesh.dx

        d_rhs_d_sol_prim = d_flux_d_sol_prim.copy()

        # Contribution to main block diagonal from source term Jacobian
        if not solver.source_off:
            d_source_d_sol_prim = self.reaction_model.calc_jacob_prim(sol_int, samp_idxs=self.direct_samp_idxs)
            d_rhs_d_sol_prim -= d_source_d_sol_prim

        # TODO: make this specific for each ImplicitIntegrator
        dt_coeff_idx = min(solver.iter, self.time_integrator.time_order) - 1
        dt_inv = self.time_integrator.coeffs[dt_coeff_idx][0] / self.time_integrator.dt

        # Modifications depending on whether dual-time integration is being used
        if self.time_integrator.dual_time:

            # Contribution to main block diagonal from solution Jacobian
            # TODO: move these conditionals into calc_adaptive_dtau(), change to calc_dtau()
            gamma_matrix = sol_int.calc_d_sol_cons_d_sol_prim(samp_idxs=self.direct_samp_idxs)
            if self.time_integrator.adapt_dtau:
                dtau_inv = sol_int.calc_adaptive_dtau(self.mesh)
            else:
                dtau_inv = 1.0 / self.time_integrator.dtau * np.ones(self.num_samp_cells, dtype=REAL_TYPE)

            d_rhs_d_sol_prim += gamma_matrix * (dtau_inv[None, None, :] + dt_inv)

            # Assemble sparse Jacobian from main, upper, and lower block diagonals
            res_jacob = sol_int.res_jacob_assemble(d_rhs_d_sol_prim, d_flux_d_sol_prim_left, d_flux_d_sol_prim_right)

        else:
            # TODO: this is hilariously inefficient, need to make Jacobian functions w/r/t conservative state
            # 	Convergence is also noticeably worse, since this is approximate
            # 	Transposes are due to matmul assuming stacks are in first index, maybe a better way to do this?

            gamma_matrix_inv = np.transpose(sol_int.calc_d_sol_prim_d_sol_cons(samp_idxs=self.gamma_idxs), axes=(2, 0, 1))

            d_rhs_d_sol_cons = np.transpose(
                np.transpose(d_rhs_d_sol_prim, axes=(2, 0, 1)) @ gamma_matrix_inv[self.gamma_idxs_center, :, :], axes=(1, 2, 0)
            )
            d_flux_d_sol_cons_left = np.transpose(
                np.transpose(d_flux_d_sol_prim_left, axes=(2, 0, 1)) @ gamma_matrix_inv[self.gamma_idxs_left, :, :], axes=(1, 2, 0)
            )
            d_flux_d_sol_cons_right = np.transpose(
                np.transpose(d_flux_d_sol_prim_right, axes=(2, 0, 1)) @ gamma_matrix_inv[self.gamma_idxs_right, :, :], axes=(1, 2, 0)
            )

            dt_arr = np.repeat(dt_inv * np.eye(gas.num_eqs)[:, :, None], self.num_samp_cells, axis=2)
            d_rhs_d_sol_cons += dt_arr

            res_jacob = sol_int.res_jacob_assemble(d_rhs_d_sol_cons, d_flux_d_sol_cons_left, d_flux_d_sol_cons_right)

        return res_jacob

    def write_iter_outputs(self, solver):
        """
        Helper function to save restart files and update probe/snapshot data
        """

        # Write restart files
        if solver.save_restarts and ((solver.iter % solver.restart_interval) == 0):
            self.sol_int.write_restart_file(solver)

        # Update probe data
        if self.num_probes > 0:
            self.update_probes(solver)

        # Update snapshot data (not written if running steady)
        if not solver.run_steady:
            if (solver.iter % solver.out_interval) == 0:
                self.sol_int.update_snapshots(solver)

    def write_steady_outputs(self, solver):
        """
        Helper function for write "steady" outputs and
        check "convergence" criterion
        """

        # Update convergence and field data file on disk
        if (solver.iter % solver.out_interval) == 0:
            self.sol_int.write_steady_data(solver)

        # Check for "convergence"
        break_flag = False
        if self.sol_int.d_sol_norm_l2 < solver.steady_tol:
            print("Steady solution criterion met, terminating run")
            break_flag = True

        return break_flag

    def write_final_outputs(self, solver):
        """
        Helper function to write final field and probe data to disk
        """

        if solver.solve_failed:
            solver.sim_type += "_FAILED"

        if not solver.run_steady:
            self.sol_int.write_snapshots(solver, solver.solve_failed)

        if self.num_probes > 0:
            self.write_probes(solver)

    def update_probes(self, solver):
        """
        Update probe storage
        """

        # TODO: throw error for source probe in ghost cells

        for probe_iter, probe_idx in enumerate(self.probe_idxs):

            probe_sec = self.probe_secs[probe_iter]
            if probe_sec == "inlet":
                sol_prim_probe = self.sol_inlet.sol_prim[:, 0]
                sol_cons_probe = self.sol_inlet.sol_cons[:, 0]
            elif probe_sec == "outlet":
                sol_prim_probe = self.sol_outlet.sol_prim[:, 0]
                sol_cons_probe = self.sol_outlet.sol_cons[:, 0]
            else:
                sol_prim_probe = self.sol_int.sol_prim[:, probe_idx]
                sol_cons_probe = self.sol_int.sol_cons[:, probe_idx]
                sol_source_probe = self.sol_int.source[:, probe_idx]

            probe = []
            for var_str in self.probe_vars:
                if var_str == "pressure":
                    probe.append(sol_prim_probe[0])
                elif var_str == "velocity":
                    probe.append(sol_prim_probe[1])
                elif var_str == "temperature":
                    probe.append(sol_prim_probe[2])
                elif var_str == "source":
                    probe.append(sol_source_probe[0])
                elif var_str == "density":
                    probe.append(sol_cons_probe[0])
                elif var_str == "momentum":
                    probe.append(sol_cons_probe[1])
                elif var_str == "energy":
                    probe.append(sol_cons_probe[2])
                elif var_str == "species":
                    probe.append(sol_prim_probe[3])
                elif var_str[:7] == "species":
                    spec_idx = int(var_str[7:])
                    probe.append(sol_prim_probe[3 + spec_idx - 1])
                elif var_str[:15] == "density-species":
                    spec_idx = int(var_str[15:])
                    probe.append(sol_cons_probe[3 + spec_idx - 1])
                else:
                    raise ValueError("Invalid probe variable " + str(var_str))

            self.probe_vals[probe_iter, :, solver.iter - 1] = probe

    def write_probes(self, solver):
        """
        Save probe data to disk
        """

        probe_file_base_name = "probe"
        for vis_var in self.probe_vars:
            probe_file_base_name += "_" + vis_var

        for probe_num in range(self.num_probes):

            # Account for failed simulations
            time_out = self.time_vals[: solver.iter]
            probe_out = self.probe_vals[probe_num, :, : solver.iter]

            probe_file_name = probe_file_base_name + "_" + str(probe_num + 1) + "_" + solver.sim_type + ".npy"
            probe_file = os.path.join(solver.probe_output_dir, probe_file_name)

            probe_save = np.concatenate((time_out[None, :], probe_out), axis=0)
            np.save(probe_file, probe_save)

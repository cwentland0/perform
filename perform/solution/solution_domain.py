import os

import numpy as np
from scipy.sparse.linalg import spsolve

from perform.constants import REAL_TYPE
from perform.input_funcs import get_initial_conditions, catch_list, catch_input, read_input_file, get_absolute_path
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
from perform.limiter.barth_jesp_limiter import BarthJespCellLimiter, BarthJespFaceLimiter
from perform.limiter.venkat_limiter import VenkatLimiter

# gas models
# TODO: make an __init__.py with get_gas_model()
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas

# reaction models
# TODO: make an __init__.py with get_reaction_model()
from perform.reaction_model.finite_rate_irrev_reaction import FiniteRateIrrevReaction


class SolutionDomain:
    """Container class for governing equations solver objects and methods.

    This class provides a broad container to hold just about everything related to the unsteady solution of
    the governing PDE. This includes the SolutionPhys objects of the interior cells and boundary cells,
    the Mesh, GasModel, Flux's, ReactionModel, TimeIntegrator, and Limiters needed to compute the unsteady solution,
    and any parameters which govern the solver behavior and the spatial domain.

    This class also provides broad member functions for stepping the solution forward in time, computing the non-linear
    right-hand side, the residual and residual Jacobian for implicit time integration, and other utility functions.

    Args:
        solver: SystemSolver containing global simulation parameters.

    Attributes:
        mesh: Mesh object associated with this SolutionDomain.
        gas_model: GasModel object associated with this SolutionDomain.
        reaction_model:
            ReactionModel object associated with this SolutionDomain if a reaction model is requested,
            None otherwise
        time_integrator: TimeIntegrator object associated with this SolutionDomain.
        sol_int: SolutionPhys representing the solution profile of the interior finite volume cells.
        sol_inlet: SolutionPhys representing the inlet ghost cell.
        sol_outlet: SolutionPhys representing the outlet ghost cell.
        invisc_flux_name: String name of the inviscid flux scheme to be applied to this SolutionDomain.
        invisc_flux_scheme: Flux object corresponding to the inviscid flux scheme associated with this SolutionDomain.
        sol_ave: SolutionPhys representing the average state at cell faces.
        visc_flux_name: String name of the flux scheme to be applied to this SolutionDomain.
        visc_flux_scheme:
            Flux object corresponding to the viscous flux scheme associated with this SolutionDomain
            if a viscous scheme is requested, None otherwise.
        space_order: Order of accuracy of the face reconstruction for computing numerical fluxes.
        grad_limiter_name:
            String name of the gradient limiter to be applied to this SolutionDomain, if space_order > 1.
        grad_limiter:
            Limiter object associated with this SolutionDomain if a gradient limiter is requested
            and space_order > 1.
        sol_left:
            SolutionPhys representing the solution profile of the reconstructed solution to the left of each face.
        sol_right:
            SolutionPhys representing the solution profile of the reconstructed solution to the right of each face.
        sol_prim_full:
            NumPy array of primitive solution profile including boundary ghost cell states, to avoid repeated
            concatenations for gradient calculations.
        sol_cons_full:
            NumPy array of conservative solution profile including boundary ghost cell states, to avoid repeated
            concatenations for gradient calculations.
        probe_locs: List of spatial coodinates within this SolutionDomain of probe monitors.
        probe_vars: List of strings of variables to be probed at each probe monitor location.
        num_probes: Number of probe monitors.
        num_probe_vars: Number of variables recorded by probe monitors.
        probe_vals: NumPy array of probe monitor time history for each monitored variable.
        probe_idxs: List of cell indices within Mesh interior cells where probe monitors are placed.
        probe_secs:
            List of strings indicating the SolutionDomain "section"
            (either "interior", "inlet", or "outlet") where each probe monitor is placed.
        time_vals: NumPy array of time values associated with each discrete time step, not including t = 0.

    """

    def __init__(self, solver):

        param_dict = solver.param_dict

        # Spatial domain
        mesh_file = get_absolute_path(str(param_dict["mesh_file"]), solver.working_dir)
        mesh_dict = read_input_file(mesh_file)
        self.mesh = Mesh(mesh_dict)

        # Gas model
        chem_file = get_absolute_path(str(param_dict["chem_file"]), solver.working_dir)
        chem_dict = read_input_file(chem_file)
        gas_model_name = catch_input(chem_dict, "gas_model", "cpg")
        if gas_model_name == "cpg":
            self.gas_model = CaloricallyPerfectGas(chem_dict)
        else:
            raise ValueError("Ivalid choice of gas_model: " + gas_model_name)
        gas = self.gas_model

        # Reaction model
        reaction_model_name = catch_input(chem_dict, "reaction_model", "none")
        if reaction_model_name == "none":
            assert solver.source_off, "Must provide a valid reaction_model_name if source_off = False"
            num_reactions = 0
        else:
            if reaction_model_name == "fr_irrev":
                self.reaction_model = FiniteRateIrrevReaction(gas, chem_dict)
            else:
                raise ValueError("Invalid choice of reaction_model: " + reaction_model_name)

            num_reactions = self.reaction_model.num_reactions

        # Time integrator
        self.time_integrator = get_time_integrator(solver.time_scheme, param_dict)

        # Solutions
        sol_prim_init = get_initial_conditions(self, solver)
        self.sol_int = SolutionInterior(
            gas, sol_prim_init, solver, self.mesh.num_cells, num_reactions, self.time_integrator
        )
        self.sol_inlet = SolutionInlet(gas, solver)
        self.sol_outlet = SolutionOutlet(gas, solver)

        # Flux schemes
        self.invisc_flux_name = catch_input(param_dict, "invisc_flux_scheme", "roe")
        self.visc_flux_name = catch_input(param_dict, "visc_flux_scheme", "invisc")

        # Inviscid flux scheme
        if self.invisc_flux_name == "roe":
            self.invisc_flux_scheme = RoeInviscFlux(self)
            # TODO: move this to the actual flux class
            ones_prof = np.ones((self.gas_model.num_eqs, self.sol_int.num_cells + 1), dtype=REAL_TYPE)
            self.sol_ave = SolutionPhys(gas, self.sol_int.num_cells + 1, sol_prim_in=ones_prof)
        else:
            raise ValueError("Invalid entry for invisc_flux_name: " + str(self.invisc_flux_name))

        # Viscous flux scheme
        if self.visc_flux_name == "invisc":
            pass
        elif self.visc_flux_name == "standard":
            self.visc_flux_scheme = StandardViscFlux(self)
        else:
            raise ValueError("Invalid entry for visc_flux_name: " + str(self.visc_flux_name))

        # Higher-order reconstructions and gradient limiters
        self.space_order = catch_input(param_dict, "space_order", 1)
        assert self.space_order >= 1, "space_order must be a positive integer"
        if self.space_order > 1:
            self.grad_limiter_name = catch_input(param_dict, "grad_limiter", "none")

            if self.grad_limiter_name == "none":
                pass
            elif self.grad_limiter_name == "barth":
                self.grad_limiter = BarthJespCellLimiter()
            elif self.grad_limiter_name == "barth_face":
                self.grad_limiter = BarthJespFaceLimiter()
            elif self.grad_limiter_name == "venkat":
                self.grad_limiter = VenkatLimiter()
            else:
                raise ValueError("Invalid entry for grad_limiter_name: " + str(self.grad_limiter_name))

        # For flux calculations
        ones_prof = np.ones((self.gas_model.num_eqs, self.sol_int.num_cells + 1), dtype=REAL_TYPE)
        self.sol_left = SolutionPhys(gas, self.sol_int.num_cells + 1, sol_prim_in=ones_prof)
        self.sol_right = SolutionPhys(gas, self.sol_int.num_cells + 1, sol_prim_in=ones_prof)

        # To avoid repeated concatenation of ghost cell states
        self.sol_prim_full = np.zeros(
            (self.gas_model.num_eqs, (self.sol_inlet.num_cells + self.sol_int.num_cells + self.sol_outlet.num_cells)),
            dtype=REAL_TYPE,
        )
        self.sol_cons_full = np.zeros(self.sol_prim_full.shape, dtype=REAL_TYPE)

        # Probe storage (as this can include boundaries as well)
        self.probe_locs = catch_list(param_dict, "probe_locs", [None])
        self.probe_vars = catch_list(param_dict, "probe_vars", [None])
        if (self.probe_locs[0] is not None) and (self.probe_vars[0] is not None):
            self.num_probes = len(self.probe_locs)
            self.num_probe_vars = len(self.probe_vars)
            self.probe_vals = np.zeros((self.num_probes, self.num_probe_vars, solver.num_steps), dtype=REAL_TYPE)

            # Get probe locations
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

        # Copy this for use with plotting functions
        solver.num_probes = self.num_probes
        solver.probe_vars = self.probe_vars

        # TODO: include initial conditions in probe_vals, time_vals
        self.time_vals = np.linspace(
            solver.sol_time + solver.dt * (solver.time_iter),
            solver.dt * (solver.time_iter - 1 + solver.num_steps),
            solver.num_steps,
            dtype=REAL_TYPE,
        )

        # For compatability with hyper-reduction, are overwritten if actually using hyper-reduction
        self.num_samp_cells = self.mesh.num_cells
        self.num_flux_faces = self.mesh.num_cells + 1
        self.num_grad_cells = self.mesh.num_cells
        self.direct_samp_idxs = np.arange(0, self.mesh.num_cells)
        self.flux_samp_left_idxs = np.s_[:-1]
        self.flux_samp_right_idxs = np.s_[1:]
        self.grad_idxs = np.arange(1, self.mesh.num_cells + 1)
        self.grad_neigh_idxs = np.s_[:]
        self.grad_neigh_extract = np.s_[1:-1]
        self.flux_left_extract = np.s_[1:]
        self.flux_right_extract = np.s_[:-1]
        self.grad_left_extract = np.s_[:]
        self.grad_right_extract = np.s_[:]
        self.flux_rhs_idxs = np.arange(0, self.mesh.num_cells)
        self.jacob_left_samp = self.flux_rhs_idxs[1:].copy()
        self.jacob_right_samp = self.flux_rhs_idxs[:-1].copy() + 1

        # TODO: remove these once conservative Jacobians are implemented
        self.gamma_idxs = np.s_[:]
        self.gamma_idxs_center = np.s_[:]
        self.gamma_idxs_left = np.s_[:-1]
        self.gamma_idxs_right = np.s_[1:]

    def advance_iter(self, solver):
        """Advance physical solution forward one physical time iteration.

        Outputs iterations number, and for implicit time integrators, residual norms for iterative convergence.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        if (not solver.run_steady) and solver.stdout:
            print("Iteration " + str(solver.iter))

        for self.time_integrator.subiter in range(self.time_integrator.subiter_max):

            self.advance_subiter(solver)

            # Check iterative solver convergence
            if self.time_integrator.time_type == "implicit":
                self.sol_int.calc_res_norms(solver, self.time_integrator.subiter)

                if self.sol_int.res_norm_l2 < self.time_integrator.res_tol:
                    break

        # Check "steady" convergence
        if solver.run_steady:
            self.sol_int.calc_d_sol_norms(solver, self.time_integrator.time_type)

        self.sol_int.update_sol_hist()

    def advance_subiter(self, solver):
        """Advance physical solution forward one subiteration of time integrator.

        For implicit time integrator, computes RHS, residual, and residual Jacobian, then computes Newton iteration.
        Linear solve residual is saved for computing convergence norms later.

        For explicit time integrator, only computes RHS and advances solution explicitly.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        self.calc_rhs(solver)

        sol_int = self.sol_int
        gas_model = self.gas_model
        mesh = self.mesh

        if self.time_integrator.time_type == "implicit":

            res = self.time_integrator.calc_residual(sol_int.sol_hist_cons, sol_int.rhs, solver)
            res_jacob = self.calc_res_jacob(solver)

            d_sol = spsolve(res_jacob, res.ravel("C"))

            # If solving in dual time, solving for primitive state
            if self.time_integrator.dual_time:
                sol_int.sol_prim += d_sol.reshape((gas_model.num_eqs, mesh.num_cells), order="C")
            else:
                sol_int.sol_cons += d_sol.reshape((gas_model.num_eqs, mesh.num_cells), order="C")

            sol_int.update_state(from_prim=self.time_integrator.dual_time)
            sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
            sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

            # Use sol_int.res to store linear solve residual
            # TODO: this should be separate for later plotting and output
            res = res_jacob @ d_sol - res.ravel("C")
            sol_int.res = np.reshape(res, (gas_model.num_eqs, mesh.num_cells), order="C")

        else:

            d_sol = self.time_integrator.solve_sol_change(sol_int.rhs)
            sol_int.sol_cons = sol_int.sol_hist_cons[0] + d_sol
            sol_int.update_state(from_prim=False)

    def calc_rhs(self, solver):
        """Compute non-linear right-hand side function of spatially-discrete governing ODE.

        Computes boundary ghost cell states, preforms higher-order face reconstructions if requested,
        computes face fluxes and source term, and stores result in sol_int.rhs.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        sol_int = self.sol_int
        samp_idxs = self.direct_samp_idxs

        # Compute ghost cell state
        self.calc_ghost_cells(solver)

        # For gradient and viscous flux calculations
        self.fill_sol_full()

        # Compute face reconstructions
        self.calc_face_recon()

        # Compute average state at face
        self.invisc_flux_scheme.calc_avg_state(self.sol_left, self.sol_right, self.sol_ave)

        # Compute fluxes
        flux = self.invisc_flux_scheme.calc_flux(self)
        if self.visc_flux_name != "invisc":
            flux -= self.visc_flux_scheme.calc_flux(self)

        # Compute maximum wave speeds
        if self.time_integrator.adapt_dtau:
            srf = np.maximum(self.sol_ave.sol_prim[1, :] + self.sol_ave.c, self.sol_ave.sol_prim[1, :] - self.sol_ave.c)
            self.sol_int.srf[samp_idxs] = np.maximum(srf[self.flux_rhs_idxs], srf[self.flux_rhs_idxs + 1])

        # Compute RHS
        self.sol_int.rhs[:, samp_idxs] = flux[:, self.flux_rhs_idxs] - flux[:, self.flux_rhs_idxs + 1]
        sol_int.rhs[:, samp_idxs] /= self.mesh.dx

        # Compute source term
        if not solver.source_off:
            reaction_source, wf, heat_release = self.reaction_model.calc_reaction(
                self.sol_int, solver.dt, samp_idxs=samp_idxs
            )

            sol_int.reaction_source[:, samp_idxs] = reaction_source
            sol_int.wf[:, samp_idxs] = wf
            sol_int.heat_release[samp_idxs] = heat_release

            sol_int.rhs[3:, samp_idxs] += sol_int.reaction_source[:-1, samp_idxs]

    def calc_ghost_cells(self, solver):
        """Calculate state in inlet and outlet boundary ghost cells.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        sol_int = self.sol_int

        if self.direct_samp_idxs[0] == 0:
            self.sol_inlet.calc_boundary_state(
                solver.sol_time, self.space_order, sol_prim=sol_int.sol_prim[:, :2], sol_cons=sol_int.sol_cons[:, :2]
            )

        if self.direct_samp_idxs[-1] == (self.mesh.num_cells - 1):
            self.sol_outlet.calc_boundary_state(
                solver.sol_time, self.space_order, sol_prim=sol_int.sol_prim[:, -2:], sol_cons=sol_int.sol_cons[:, -2:]
            )

    def fill_sol_full(self):
        """Fill sol_prim_full and sol_cons_full from interior and ghost cells"""

        # TODO: hyper-reduction indices

        idx_in = self.sol_inlet.num_cells
        idx_int = idx_in + self.sol_int.num_cells

        self.sol_prim_full[:, :idx_in] = self.sol_inlet.sol_prim.copy()
        self.sol_prim_full[:, idx_in:idx_int] = self.sol_int.sol_prim.copy()
        self.sol_prim_full[:, idx_int:] = self.sol_outlet.sol_prim.copy()

        self.sol_cons_full[:, :idx_in] = self.sol_inlet.sol_cons.copy()
        self.sol_cons_full[:, idx_in:idx_int] = self.sol_int.sol_cons.copy()
        self.sol_cons_full[:, idx_int:] = self.sol_outlet.sol_cons.copy()

    def calc_face_recon(self):
        """Calculate reconstruction of state at finite volume faces.

        For first-order scheme, simply use cell-centered values.
        For higher-order, use gradient to compute reconstruction.
        """

        # First order
        self.sol_left.sol_prim[:, :] = self.sol_prim_full[:, self.flux_samp_left_idxs]
        self.sol_left.sol_cons[:, :] = self.sol_cons_full[:, self.flux_samp_left_idxs]
        self.sol_right.sol_prim[:, :] = self.sol_prim_full[:, self.flux_samp_right_idxs]
        self.sol_right.sol_cons[:, :] = self.sol_cons_full[:, self.flux_samp_right_idxs]

        # Higher order
        if self.space_order > 1:

            grad = self.calc_cell_gradients(self.sol_prim_full)
            sol_left = self.sol_left.sol_prim
            sol_right = self.sol_right.sol_prim

            self.add_high_order_contrib(sol_left, grad, self.grad_left_extract, self.flux_left_extract, side="left")

            self.add_high_order_contrib(
                sol_right,
                grad,
                self.grad_right_extract,
                self.flux_right_extract,
                side="right",
            )

            # TODO: the following fails miserably for dual_time = False and grad_limiter = "barth" need to check this

            # if self.time_integrator.dual_time:
            #     grad = self.calc_cell_gradients(self.sol_prim_full)
            #     sol_left = self.sol_left.sol_prim
            #     sol_right = self.sol_right.sol_prim
            # else:
            #     grad = self.calc_cell_gradients(self.sol_cons_full)
            #     sol_left = self.sol_left.sol_cons
            #     sol_right = self.sol_right.sol_cons

            # self.add_high_order_contrib(
            #     sol_left, grad, self.grad_left_extract, self.flux_left_extract, side="left"
            # )

            # self.add_high_order_contrib(
            #     sol_right, grad, self.grad_right_extract, self.flux_right_extract, side="right",
            # )

        # update state and thermo properties
        self.sol_left.update_state(from_prim=True)
        self.sol_right.update_state(from_prim=True)

        # TODO: switch above to below when you figure out the issue with dual_time=False and Barth limiter
        # self.sol_left.update_state(from_prim=self.time_integrator.dual_time)
        # self.sol_right.update_state(from_prim=self.time_integrator.dual_time)

    def add_high_order_contrib(self, sol, grad, grad_samp_idxs, face_grad_idxs, side):
        """Adds contribution of higher-order gradient calculations to face reconstruction.

        Args:
            sol: NumPy array of first-order solutions at left or right side of cell faces.
            grad: NumPy array of cell-centered gradients.
            grad_samp_idxs: NumPy array of cell indices at which to subsample grad for hyper-reduction.
            face_grad_idxs: NumPy array of face indices at which to subsample sol for hyper-reduciton.
            side: string; if "left", compute for face's left side. If "right", compute for face's right side.
        """

        sol_change = (self.mesh.dx / 2.0) * grad[:, grad_samp_idxs]
        if side == "left":
            sol[:, face_grad_idxs] += sol_change
        elif side == "right":
            sol[:, face_grad_idxs] -= sol_change

    def calc_cell_gradients(self, sol_full):
        """Compute cell-centered gradients and gradient limiters for higher-order face reconstructions.

        Calculates cell-centered gradients via a finite-difference stencil. If a gradient limiter is requested,
        the multiplicative limiter factor is computed and the gradient is multiplied by this factor.

        Args:
            sol_full: NumPy array of cell-centered states, including ghost cells.

        Returns:
            NumPy array of cell-centered gradients of state profile at all interior cells.
        """

        # Compute gradients via finite difference stencil
        grad = np.zeros((self.gas_model.num_eqs, self.num_grad_cells), dtype=REAL_TYPE)
        if self.space_order == 2:
            grad = (0.5 / self.mesh.dx) * (sol_full[:, self.grad_idxs + 1] - sol_full[:, self.grad_idxs - 1])
        else:
            raise ValueError("Order " + str(self.space_order) + " gradient calculations not implemented")

        # Compute gradient limiter and limit gradient, if requested
        if self.grad_limiter_name != "none":
            phi = self.grad_limiter.calc_limiter(self, sol_full, grad)
            grad *= phi

        return grad

    def calc_res_jacob(self, solver):
        """Compute Jacobian of full-discrete residual.

        Calculates the gradient of the fully-discrete (i.e. after numerical time integration) residual vector
        with respect to either the full primitive state vector (if dual_time == True) or the full conservative
        state vector (if dual_time == False). The contributions to the residual Jacobian from the time integrator,
        flux, and source terms are computed separately and combined.

        Ultimately, the residual Jacobian can be represented as a block tri-diagonal matrix
        if the residual vector were flattened in column-major ordering. For the sake of computational efficiency,
        however, it is later formed into a roughly banded matrix corresponding to the residual vector being flattened
        in row-major ordering. This final reordering is performed via res_jacob_assemble().

        Args:
            solver: SystemSolver containing global simulation parameters.

        Returns:
            2D scipy.sparse.csr_matrix of the residual Jacobian, ordered in such a way as to correspond with a
            row-major flattening of the residual vector (i.e. first by cells, then by variables).
        """

        sol_int = self.sol_int
        gas = self.gas_model
        samp_idxs = self.gamma_idxs

        # Calculate RHS and solution Jacobians
        sol_jacob = sol_int.calc_sol_jacob(not self.time_integrator.dual_time, samp_idxs=samp_idxs)
        rhs_jacob_center, flux_jacob_left, flux_jacob_right = self.calc_rhs_jacob(solver)

        # TODO: make this specific for each ImplicitIntegrator
        dt_coeff_idx = min(self.time_integrator.cold_start_iter, self.time_integrator.time_order) - 1
        # dt_coeff_idx = min(solver.iter, self.time_integrator.time_order) - 1
        dt_inv = self.time_integrator.coeffs[dt_coeff_idx][0] / self.time_integrator.dt

        # Modifications depending on whether dual-time integration is being used
        if self.time_integrator.dual_time:

            dtau = self.calc_dtau()
            rhs_jacob_center += sol_jacob * (1.0 / dtau[None, None, :] + dt_inv)

            # Assemble sparse Jacobian from main, upper, and lower block diagonals
            res_jacob = sol_int.res_jacob_assemble(rhs_jacob_center, flux_jacob_left, flux_jacob_right)

        else:
            # TODO: this is hilariously inefficient, need to make Jacobian functions w/r/t conservative state
            # 	Convergence is also noticeably worse, since this is approximate
            # 	Transposes are due to matmul assuming stacks are in first index, maybe a better way to do this?

            sol_jacob = np.transpose(sol_jacob, axes=(2, 0, 1))

            rhs_jacob_center = np.transpose(
                np.transpose(rhs_jacob_center, axes=(2, 0, 1)) @ sol_jacob[self.gamma_idxs_center, :, :], axes=(1, 2, 0)
            )
            flux_jacob_left = np.transpose(
                np.transpose(flux_jacob_left, axes=(2, 0, 1)) @ sol_jacob[self.gamma_idxs_left, :, :], axes=(1, 2, 0)
            )
            flux_jacob_right = np.transpose(
                np.transpose(flux_jacob_right, axes=(2, 0, 1)) @ sol_jacob[self.gamma_idxs_right, :, :], axes=(1, 2, 0)
            )

            dt_arr = np.repeat(dt_inv * np.eye(gas.num_eqs)[:, :, None], self.num_samp_cells, axis=2)
            rhs_jacob_center += dt_arr

            res_jacob = sol_int.res_jacob_assemble(rhs_jacob_center, flux_jacob_left, flux_jacob_right)

        return res_jacob

    def calc_rhs_jacob(self, solver):
        """Compute Jacobian of the right-hand side of the semi-discrete governing ODE.

        Calculates and collects contributions from the inviscid flux, viscous flux, and reaction source term.
        If dual_time is True, computes the Jacobians with respect to the primitive variables.
        Otherwise computes the Jacobians with respect to the conservative variables.

        Args:
            solver: SystemSolver containing global simulation parameters.

        Returns:
            rhs_jacob_center: center block diagonal of flux and source contributions to the Jacobian, representing
            the gradient of a given cell's contribution with respect to its own state.
            flux_jacob_left: lower block diagonal of flux Jacobian, representing the gradient of a given cell's
            flux contribution with respect to its left neighbor's state.
            flux_jacob_right: upper block diagonal of flux Jacobian, representing the gradient of a given cell's
            flux contribution with respect to its right neighbor's state.
        """

        sol_int = self.sol_int

        # Flux jacobians
        flux_jacob_center, flux_jacob_left, flux_jacob_right = self.invisc_flux_scheme.calc_jacob(
            self, wrt_prim=self.time_integrator.dual_time
        )

        if self.visc_flux_name != "invisc":
            visc_flux_jacob_center, visc_flux_jacob_left, visc_flux_jacob_right = self.visc_flux_scheme.calc_jacob(
                self, wrt_prim=self.time_integrator.dual_time
            )

            flux_jacob_center += visc_flux_jacob_center
            flux_jacob_left += visc_flux_jacob_left
            flux_jacob_right += visc_flux_jacob_right

        flux_jacob_center /= self.mesh.dx
        flux_jacob_left /= self.mesh.dx
        flux_jacob_right /= self.mesh.dx

        rhs_jacob_center = flux_jacob_center.copy()

        # Contribution to main block diagonal from source term Jacobian
        if not solver.source_off:
            source_jacob = self.reaction_model.calc_jacob(
                sol_int, wrt_prim=self.time_integrator.dual_time, samp_idxs=self.direct_samp_idxs
            )
            rhs_jacob_center -= source_jacob

        return rhs_jacob_center, flux_jacob_left, flux_jacob_right

    def calc_dtau(self):
        """Calculate dtau for each cell.

        If adapt_dtau is True, will adapt dtau based on user input constraints and local wave speed.
        Otherwise, will simply return the fixed value of dtau.

        Adaptation is intended to improve dual time-stepping robustness, but mostly acts to slow convergence.
        For now, I recommend not setting adapt_dtau to False until this is completed.

        Returns:
            NumPy array of the dtau profile.
        """

        # If not adapting, just return fixed dtau
        if not self.time_integrator.adapt_dtau:
            dtau = self.time_integrator.dtau * np.ones(self.num_samp_cells, dtype=REAL_TYPE)
            return dtau

        sol_int = self.sol_int
        samp_idxs = self.direct_samp_idxs

        # Compute initial dtau from input cfl and srf
        dtaum = 1.0 * self.mesh.dx / self.sol_int.srf[samp_idxs]
        dtau = self.time_integrator.cfl * dtaum

        # Limit by von Neumann number
        if self.visc_flux_name != "invisc":
            # TODO: calculating this is stupidly expensive
            kin_visc_mix = sol_int.dyn_visc_mix / sol_int.sol_cons[0, samp_idxs]
            dtau = np.minimum(dtau, self.time_integrator.vnn * np.square(self.mesh.dx) / kin_visc_mix)
            dtaum = np.minimum(dtaum, 3.0 / kin_visc_mix)

        # Limit dtau
        # TODO: finish implementation

        return dtau

    def write_iter_outputs(self, solver):
        """Store and save data at time step intervals.

        Write restart files at the time step interval given by solver.restart_interval.
        Update probe data arrays at every time step.
        Update snapshot array at interval given by solver.out_interval if not running in "steady" mode.

        Args:
            solver: SystemSolver containing global simulation parameters.
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

            # write intermediate snapshots, if requested
            if solver.out_itmdt_interval is not None:
                if (solver.iter % solver.out_itmdt_interval) == 0:
                    self.sol_int.write_snapshots(solver, intermediate=True, failed=False)

        # write intermediate probes, if requested
        if solver.out_itmdt_interval is not None:
            if ((solver.iter % solver.out_itmdt_interval) == 0) and (self.num_probes > 0):
                self.write_probes(solver, intermediate=True, failed=False)

    def write_steady_outputs(self, solver):
        """Saves "steady" solver outputs and check "convergence" criterion.

        When running in "steady" mode, the solution gets written to disk at the interval given by solver.out_interval.
        Additionally, the norm of the difference between the current and previous solution is checked to determine
        whether the "steady" solve has "converged" according to solver.steady_tol.

        Args:
            solver: SystemSolver containing global simulation parameters.

        Returns:
            Boolean flag indicating whether the "steady" solve has "converged".
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
        """Save final field and probe data to disk at end of simulation

        If simulation ran to completion, full field snapshot and probe data are written to disk.
        If the solver exploded before completing, the field snapshots and and probe data up to that point
        are written to disk.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        if not solver.run_steady:
            self.sol_int.write_snapshots(solver, intermediate=False, failed=solver.solve_failed)

        if self.num_probes > 0:
            self.write_probes(solver, intermediate=False, failed=solver.solve_failed)

        # clean up intermediate results, if any
        if solver.out_itmdt_interval is not None:
            if (solver.iter / solver.out_itmdt_interval) > 1:
                self.delete_itmdt_probes(solver)
                self.sol_int.delete_itmdt_snapshots(solver)

    def update_probes(self, solver):
        """Update probe monitor array from current solution.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        for probe_iter, probe_idx in enumerate(self.probe_idxs):

            # Determine where the probe monitor is
            probe_sec = self.probe_secs[probe_iter]
            if probe_sec == "inlet":
                sol_prim_probe = self.sol_inlet.sol_prim[:, 0]
                sol_cons_probe = self.sol_inlet.sol_cons[:, 0]
                mass_fracs_full = self.sol_inlet.mass_fracs_full[:, 0]
            elif probe_sec == "outlet":
                sol_prim_probe = self.sol_outlet.sol_prim[:, 0]
                sol_cons_probe = self.sol_outlet.sol_cons[:, 0]
                mass_fracs_full = self.sol_outlet.mass_fracs_full[:, 0]
            else:
                sol_prim_probe = self.sol_int.sol_prim[:, probe_idx]
                sol_cons_probe = self.sol_int.sol_cons[:, probe_idx]
                sol_source_probe = self.sol_int.reaction_source[:, probe_idx]
                sol_hr_probe = self.sol_int.heat_release[probe_idx]
                mass_fracs_full = self.sol_int.mass_fracs_full[:, probe_idx]

            # Gather probe monitor data
            probe = []
            for var_str in self.probe_vars:
                if var_str == "pressure":
                    probe.append(sol_prim_probe[0])
                elif var_str == "velocity":
                    probe.append(sol_prim_probe[1])
                elif var_str == "temperature":
                    probe.append(sol_prim_probe[2])
                elif var_str == "density":
                    probe.append(sol_cons_probe[0])
                elif var_str == "momentum":
                    probe.append(sol_cons_probe[1])
                elif var_str == "energy":
                    probe.append(sol_cons_probe[2])
                elif var_str == "species":
                    probe.append(sol_prim_probe[3])
                elif var_str[:7] == "species":
                    spec_idx = int(var_str[8:])
                    probe.append(mass_fracs_full[spec_idx])
                elif var_str[:15] == "density-species":
                    spec_idx = int(var_str[16:])
                    if spec_idx == (self.gas_model.num_species_full - 1):
                        probe.append(mass_fracs_full[-1] * sol_cons_probe[0])
                    else:
                        probe.append(sol_cons_probe[3 + spec_idx])
                elif var_str[:6] == "source":
                    spec_idx = int(var_str[7:])
                    probe.append(sol_source_probe[spec_idx])
                elif var_str == "heat-release":
                    probe.append(sol_hr_probe)
                else:
                    raise ValueError("Invalid probe variable " + str(var_str))

            # Insert into probe data array
            self.probe_vals[probe_iter, :, solver.iter - 1] = probe

    def write_probes(self, solver, intermediate=False, failed=False):
        """Save probe data to disk at end/failure of simulation.

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        assert not (
            intermediate and failed
        ), "Something went wrong, tried to write intermediate and failed snapshots at same time"

        # Get output file name
        probe_file_base_name = "probe"
        for vis_var in self.probe_vars:
            probe_file_base_name += "_" + vis_var

        suffix = solver.sim_type
        if intermediate:
            suffix += "_ITMDT"
        elif failed:
            suffix += "_FAILED"

        for probe_num in range(self.num_probes):

            # Account for failed simulations
            time_out = self.time_vals[: solver.iter]
            probe_out = self.probe_vals[probe_num, :, : solver.iter]

            probe_file_name = probe_file_base_name + "_" + str(probe_num + 1) + "_" + suffix + ".npy"
            probe_file = os.path.join(solver.probe_output_dir, probe_file_name)

            probe_save = np.concatenate((time_out[None, :], probe_out), axis=0)
            np.save(probe_file, probe_save)

    def delete_itmdt_probes(self, solver):
        """Delete intermediate probe data

        Args:
            solver: SystemSolver containing global simulation parameters.
        """

        # Get output file name
        probe_file_base_name = "probe"
        for vis_var in self.probe_vars:
            probe_file_base_name += "_" + vis_var

        suffix = solver.sim_type + "_ITMDT"
        for probe_num in range(self.num_probes):
            probe_file_name = probe_file_base_name + "_" + str(probe_num + 1) + "_" + suffix + ".npy"
            probe_file = os.path.join(solver.probe_output_dir, probe_file_name)
            os.remove(probe_file)

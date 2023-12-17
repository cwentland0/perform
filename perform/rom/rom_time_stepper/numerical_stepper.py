import numpy as np

from perform.constants import REAL_TYPE
from perform.input_funcs import catch_input
from perform.rom.rom_time_stepper.rom_time_stepper import RomTimeStepper
import sys


class NumericalStepper(RomTimeStepper):
    def __init__(self, sol_domain, rom_domain, solver):

        # use definition of time integrator in solver_params.inp
        # NOTE: user classes are mutable, so changes to NumericalStepper.time_integrator are reflected in
        # sol_domain.time_integrator
        self.time_integrator = sol_domain.time_integrator

        super().__init__()

    def init_state(self, sol_domain, rom_domain):

        # Whether to encode initial condition lookback for higher-order methods
        # NOTE: this can be bad if the lookback time steps are poorly represented by the trial basis,
        # and may be better to just cold start the time integration
        encode_higher_order = catch_input(rom_domain.rom_dict, "encode_higher_order", False)
        if encode_higher_order:
            sol_hist = rom_domain.var_mapping.get_variable_hist_from_state_hist(sol_domain)
        else:
            sol = rom_domain.var_mapping.get_variables_from_state(sol_domain)
            # this will be a cold start
            self.time_integrator.cold_start_iter = 1

        for rom_model in rom_domain.model_list:

            # Initialize model full-order state, latent state, and histories
            if encode_higher_order:
                sol_hist_in = [sol[rom_model.var_idxs, :] for sol in sol_hist]
                code_hist_out, sol_hist_encoded_out = rom_model.space_mapping.encode_decode_series(sol_hist_in)
                rom_model.code[:] = code_hist_out[0].copy()
                rom_model.code_hist = [code.copy() for code in code_hist_out]
                rom_model.sol[:, :] = sol_hist_encoded_out[0].copy()
                rom_model.sol_hist = [sol.copy() for sol in sol_hist_encoded_out]
            else:
                sol_in = sol[rom_model.var_idxs, :]
                code, sol_encoded = rom_model.space_mapping.encode_decode_series(sol_in)
                rom_model.code[:] = code[0].copy()
                rom_model.code_hist = [code[0].copy() for _ in range(self.time_integrator.time_order + 1)]
                rom_model.sol[:, :] = sol_encoded[0].copy()
                rom_model.sol_hist = [sol_encoded[0].copy() for _ in range(self.time_integrator.time_order + 1)]

            # Implicit solve code residual
            if self.time_integrator.time_type == "implicit":
                rom_model.res = np.zeros(rom_model.latent_dim, dtype=REAL_TYPE)

        # update SolutionDomain state and history for intrusive methods
        if rom_domain.rom_method.is_intrusive:
            rom_domain.var_mapping.update_full_state(sol_domain, rom_domain)
            rom_domain.var_mapping.update_state_hist(sol_domain, rom_domain)

    def advance_iter(self, sol_domain, solver, rom_domain):   

        for self.time_integrator.subiter in range(self.time_integrator.subiter_max):

            # write non-solution outputs, which are computed at previous time step
            if self.time_integrator.subiter == 0:
                sol_domain.write_nonsol_outputs(solver)

            self.advance_subiter(sol_domain, solver, rom_domain)

            # Check iterative solver convergence
            if self.time_integrator.time_type == "implicit":
                self.calc_code_res_norms(sol_domain, solver, rom_domain)

                if sol_domain.sol_int.res_norm_l2 < self.time_integrator.res_tol:
                    break

    def advance_subiter(self, sol_domain, solver, rom_domain):
        """Advance low-dimensional state and full solution forward one subiteration of numerical time integrator.

        For intrusive ROMs, computes RHS and RHS Jacobian (if necessary).

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """

        sol_int = sol_domain.sol_int
        res, res_jacob = None, None

        if rom_domain.rom_method.is_intrusive:
            sol_domain.calc_rhs(solver)

        if self.time_integrator.time_type == "implicit":

            # Compute residual and residual Jacobian
            if rom_domain.rom_method.is_intrusive:
                res = self.time_integrator.calc_residual(
                    sol_int.sol_hist_cons, sol_int.rhs, solver, samp_idxs=sol_domain.direct_samp_idxs
                )
                res_jacob = sol_domain.calc_res_jacob(solver)

            # Compute change in low-dimensional state
            code_lhs, code_rhs = rom_domain.rom_method.calc_d_code(res_jacob, res, sol_domain, rom_domain)
            d_code = np.linalg.solve(code_lhs, code_rhs)
            res_solve = code_lhs @ d_code - code_rhs

            # Update model internal state
            latent_dim_idx = 0
            for model in rom_domain.model_list:

                # Low-dimensional quantities
                model.d_code[:] = d_code[latent_dim_idx : latent_dim_idx + model.latent_dim]
                model.res[:] = res_solve[latent_dim_idx : latent_dim_idx + model.latent_dim]

                model.code += model.d_code
                model.code_hist[0] = model.code.copy()

                # Full-dimensional quantities
                model.sol = model.space_mapping.decode_sol(model.code)
                model.sol_hist[0] = model.sol.copy()

                latent_dim_idx += model.latent_dim

        else:

            for model in rom_domain.model_list:

                model.calc_rhs_low_dim(rom_domain, sol_domain)
                d_code = self.time_integrator.solve_sol_change(model.rhs_low_dim)
                model.code = model.code_hist[0] + d_code

                # Full-dimensional quantities
                model.sol = model.space_mapping.decode_sol(model.code)
                model.sol_hist[0] = model.sol.copy()

        # Update SolutionDomain state
        rom_domain.var_mapping.update_full_state(sol_domain, rom_domain)

        # If intrusive method, update primitive and conservative history
        if rom_domain.rom_method.is_intrusive:
            sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
            sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

    def calc_code_res_norms(self, sol_domain, solver, rom_domain):
        """Calculate and print low-dimensional linear solve residual norms.

        Computes L2 and L1 norms of low-dimensional linear solve residuals for each RomModel,
        as computed in advance_subiter(). These are averaged across all RomModels and printed to the terminal,
        and are used in advance_iter() to determine whether the Newton's method iterative solve has
        converged sufficiently. If the norm is below numerical precision, it defaults to 1e-16.

        Note that terminal output is ORDER OF MAGNITUDE (i.e. 1e-X, where X is the order of magnitude).

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """

        # Compute residual norm for each model
        norm_l2_sum = 0.0
        norm_l1_sum = 0.0
        for rom_model in rom_domain.model_list:
            norm_l2, norm_l1 = self.calc_code_norms(rom_model)
            norm_l2_sum += norm_l2
            norm_l1_sum += norm_l1

        # Average over all models
        norm_l2 = norm_l2_sum / rom_domain.num_models
        norm_l1 = norm_l1_sum / rom_domain.num_models

        # Norm is sometimes zero, just default to -16 I guess
        if norm_l2 == 0.0:
            norm_out_l2 = -16.0
        else:
            norm_out_l2 = np.log10(norm_l2)

        if norm_l1 == 0.0:
            norm_out_l1 = -16.0
        else:
            norm_out_l1 = np.log10(norm_l1)

        # Print to terminal
        out_string = (str(self.time_integrator.subiter + 1) + ":\tL2: %18.14f, \tL1: %18.14f") % (
            norm_out_l2,
            norm_out_l1,
        )
        if solver.stdout:
            print(out_string)

        sol_domain.sol_int.res_norm_l2 = norm_l2
        sol_domain.sol_int.res_norm_l1 = norm_l1
        sol_domain.sol_int.res_norm_hist[solver.iter - 1, :] = [norm_l2, norm_l1]

    def calc_code_norms(self, rom_model):
        """Compute L1 and L2 norms of low-dimensional state linear solve residuals

        This function is called within RomDomain.calc_code_res_norms(), after which the residual norms are averaged
        across all models for an aggregate measure. Note that this measure is scaled by number of elements,
        so "L2 norm" here is really RMS.

        Returns:
            The L2 and L1 norms of the low-dimensional linear solve residual.
        """

        res_abs = np.abs(rom_model.res)

        # L2 norm
        res_norm_l2 = np.sum(np.square(res_abs))
        res_norm_l2 /= rom_model.latent_dim
        res_norm_l2 = np.sqrt(res_norm_l2)

        # L1 norm
        res_norm_l1 = np.sum(res_abs)
        res_norm_l1 /= rom_model.latent_dim

        return res_norm_l2, res_norm_l1

    def calc_fullydiscrhs(self, sol_domain, stateArg, solver, rom_domain, samp_idxs=np.s_[:]):
        """Compute fully discrete rhs
            stateArg is a column vector. also returns a column vector
        """
        
        assert self.time_order < 2, "BDF order has to be 1, backward Euler only"
        
        # make a deep copy of sol_domain 
        copy_sol_domain = copy.deepcopy(sol_domain)
        
        # reshape stateArg (solution reconstruction by ROM)
        stateArg_reshape = stateArg.reshape((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), order="C")
        
        # update sol_cons and sol_prim
        if sol_domain.time_integrator.dual_time:
            copy_sol_domain.sol_int.sol_prim = stateArg_reshape
        else:
            copy_sol_domain.sol_int.sol_cons = stateArg_reshape
        copy_sol_domain.sol_int.update_state(from_cons=(not sol_domain.time_integrator.dual_time))
        
        # update deim indices
        copy_sol_domain.direct_samp_idxs = np.arange(0, sol_domain.mesh.num_cells)
        copy_sol_domain.num_samp_cells = len(copy_sol_domain.direct_samp_idxs)
        rom_domain.compute_cellidx_hyper_reduc(copy_sol_domain)
        
        # compute rhs 
        copy_sol_domain.calc_rhs(solver)
        rhs_main = copy_sol_domain.sol_int.rhs # shape will be num_eqs x num_cells
        
        # reshape rhs to be a column vector
        rhs = rhs_main.reshape((-1,1))
        
        # calculate semi-discrete rhs here
        # Account for cold start
        time_order = min(solver.iter, self.time_order)

        coeffs = self.coeffs[time_order - 1]

        # compute the rhs function according to the time advancement scheme
        if sol_domain.time_integrator.dual_time:
            res_jacob = copy_sol_domain.calc_res_jacob(solver)
            residual = coeffs[0] * copy_sol_domain.sol_int.sol_hist_cons[0][:, samp_idxs]
            for iter_idx in range(1, time_order + 1):
                residual += coeffs[iter_idx] * copy_sol_domain.sol_int.sol_hist_cons[iter_idx][:, samp_idxs]
            residual = -(residual / self.dt) + rhs_main[:, samp_idxs]
            d_sol = spsolve(res_jacob, residual.ravel("C"))
            fullydiscrhs = stateArg[samp_idxs, :] + np.expand_dims(d_sol, axis=1)
        else:
            fullydiscrhs = coeffs[0] * stateArg[samp_idxs, :] + self.dt * rhs[samp_idxs, :] 
        
        return fullydiscrhs


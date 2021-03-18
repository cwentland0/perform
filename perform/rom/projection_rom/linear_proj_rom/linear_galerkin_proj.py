import numpy as np

from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearGalerkinProj(LinearProjROM):
    """
    Class for linear decoder and Galerkin projection

    Trial basis is assumed to represent the conserved variables
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if ((rom_domain.time_integrator.time_type == "implicit")
                and (rom_domain.time_integrator.dual_time)):
            raise ValueError("Galerkin is intended for conservative"
                             + " variable evolution, please set"
                             + " dual_time = False")

        super().__init__(model_idx, rom_domain, sol_domain)

    def calc_projector(self, sol_domain):
        """
        Compute rhs projection operator
        """

        if self.hyper_reduc:
            # V^T * U * [S^T * U]^+
            self.projector = (
                self.trial_basis.T @ self.hyper_reduc_basis
                @ np.linalg.pinv(self.hyper_reduc_basis[
                    self.direct_hyper_reduc_samp_idxs, :])
            )

        else:
            # V^T
            self.projector = self.trial_basis.T

    def calc_d_code(self, res_jacob, res, sol_domain):
        """
        Compute change in low-dimensional state for
        implicit scheme Newton iteration
        """

        # TODO: should be calculated once
        scaled_trial_basis = (
            self.trial_basis
            * self.norm_fac_prof_cons.ravel(order="C")[:, None])

        # TODO: using res_jacob.toarray(), otherwise this
        # 	operation returns type np.matrix, which is undesirable
        # 	Need to figure out a more efficient method, if possible
        lhs = (
            self.trial_basis.T
            @ (res_jacob.toarray()
            / self.norm_fac_prof_cons.ravel(order="C")[:, None])
            @ scaled_trial_basis)

        rhs = (
            self.trial_basis.T
            @ (res / self.norm_fac_prof_cons).ravel(order="C"))

        d_code = np.linalg.solve(lhs, rhs)

        return d_code, lhs, rhs

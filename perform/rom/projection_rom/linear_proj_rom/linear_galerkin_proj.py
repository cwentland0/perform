import numpy as np

from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearGalerkinProj(LinearProjROM):
    """
    Class for linear decoder and Galerkin projection

    Trial basis is assumed to represent the conserved variables
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if (rom_domain.time_integrator.time_type == "implicit") and (rom_domain.time_integrator.dual_time):
            raise ValueError("Galerkin is intended for conservative variable evolution, please set dual_time = False")

        super().__init__(model_idx, rom_domain, sol_domain)

        if not rom_domain.time_integrator.time_type == "explicit":
            # precompute scaled trial basis
            self.trial_basis_scaled = self.trial_basis * self.norm_fac_prof_cons.ravel(order="C")[:, None]

        if self.hyper_reduc:
            # procompute hyper-reduction projector, V^T * U * [S^T * U]^+
            self.hyper_reduc_operator = (
                self.trial_basis.T
                @ self.hyper_reduc_basis
                @ np.linalg.pinv(self.hyper_reduc_basis[self.direct_samp_idxs_flat, :])
            )

    def calc_projector(self, sol_domain):
        """
        Compute RHS projection operator
        """

        if self.hyper_reduc:
            self.projector = self.hyper_reduc_operator
        else:
            self.projector = self.trial_basis.T

    def calc_d_code(self, res_jacob, res, sol_domain):
        """
        Compute change in low-dimensional state for implicit scheme Newton iteration
        """

        lhs = (res_jacob @ self.trial_basis_scaled) / self.norm_fac_prof_cons.ravel(order="C")[self.direct_samp_idxs_flat, None]
        res_scaled = (res / self.norm_fac_prof_cons[:, sol_domain.direct_samp_idxs]).ravel(order="C")

        if self.hyper_reduc:
            lhs = self.hyper_reduc_operator @ lhs
            rhs = self.hyper_reduc_operator @ res_scaled
        else:
            lhs = self.trial_basis.T @ lhs
            rhs = self.trial_basis.T @ res_scaled

        d_code = np.linalg.solve(lhs, rhs)

        return d_code, lhs, rhs

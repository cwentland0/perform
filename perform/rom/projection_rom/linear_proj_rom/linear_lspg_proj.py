import numpy as np

from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearLSPGProj(LinearProjROM):
    """
    Class for linear decoder and least-squares Petrov-Galerkin projection
    Trial basis is assumed to represent the conserved variables
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        # I'm not going to code LSPG with explicit time integrator,
        # it's a pointless exercise
        if rom_domain.time_integrator.time_type == "explicit":
            raise ValueError(
                "LSPG with an explicit time integrator deteriorates to Galerkin, please use"
                + " Galerkin or select an implicit time integrator."
            )

        if rom_domain.time_integrator.dual_time:
            raise ValueError("LSPG is intended for conservative variable evolution, please set dual_time = False")

        super().__init__(model_idx, rom_domain, sol_domain)

        self.trial_basis_scaled = self.trial_basis * self.norm_fac_prof_cons.ravel(order="C")[:, None]

        # procompute hyper-reduction projector, [S^T * U]^+
        # TODO: may want to have an option to compute U * [S^T * U]^+
        #   The matrix can be big, but it's necessary for conservative LSPG, different least-square problem
        if self.hyper_reduc:
            self.hyper_reduc_operator = np.linalg.pinv(self.hyper_reduc_basis[self.direct_samp_idxs_flat, :])

    def calc_d_code(self, res_jacob, res, sol_domain):
        """
        Compute change in low-dimensional state for
        implicit scheme Newton iteration
        """

        # Compute test basis
        test_basis = (res_jacob @ self.trial_basis_scaled) / self.norm_fac_prof_cons.ravel(order="C")[self.direct_samp_idxs_flat, None]
        if self.hyper_reduc:
            test_basis = self.hyper_reduc_operator @ test_basis

        # lhs and rhs of Newton iteration
        lhs = test_basis.T @ test_basis

        res_scaled = (res / self.norm_fac_prof_cons[:, sol_domain.direct_samp_idxs]).ravel(order="C")
        if self.hyper_reduc:
            res_scaled = self.hyper_reduc_operator @ res_scaled
        rhs = test_basis.T @ res_scaled

        # Linear solve
        dCode = np.linalg.solve(lhs, rhs)

        # breakpoint()

        return dCode, lhs, rhs

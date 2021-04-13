import numpy as np

from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearSPLSVTProj(LinearProjROM):
    """
    Class for linear decoder and SP-LSVT formulation
    Trial basis is assumed to represent the conserved variables
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if rom_domain.time_integrator.time_type == "explicit":
            raise ValueError("Explicit SP-LSVT not implemented yet")

        if (rom_domain.time_integrator.time_type == "implicit") and (not rom_domain.time_integrator.dual_time):
            raise ValueError(
                "SP-LSVT is intended for primitive variable evolution, please use Galerkin or LSPG,"
                + " or set dual_time = True"
            )

        super().__init__(model_idx, rom_domain, sol_domain)

        self.trial_basis_scaled = self.trial_basis * self.norm_fac_prof_prim.ravel(order="C")[:, None]

        # procompute hyper-reduction projector, U * [S^T * U]^+
        if self.hyper_reduc:
            self.hyper_reduc_operator = np.linalg.pinv(self.hyper_reduc_basis[self.direct_samp_idxs_flat, :])

    def calc_d_code(self, res_jacob, res, sol_domain):
        """
        Compute change in low-dimensional state for implicit scheme Newton iteration
        """

        # compute test basis
        test_basis = (res_jacob @ self.trial_basis_scaled) / self.norm_fac_prof_cons.ravel(order="C")[self.direct_samp_idxs_flat, None]
        if self.hyper_reduc:
            test_basis = self.hyper_reduc_operator @ test_basis

        # LHS and RHS of Newton iteration
        lhs = test_basis.T @ test_basis

        res_scaled = (res / self.norm_fac_prof_cons[:, sol_domain.direct_samp_idxs]).ravel(order="C")
        if self.hyper_reduc:
            res_scaled = self.hyper_reduc_operator @ res_scaled
        rhs = test_basis.T @ res_scaled

        # linear solve
        dCode = np.linalg.solve(lhs, rhs)

        return dCode, lhs, rhs

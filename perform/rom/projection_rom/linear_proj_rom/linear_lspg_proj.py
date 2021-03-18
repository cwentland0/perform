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

    def calc_d_code(self, res_jacob, res, sol_domain):
        """
        Compute change in low-dimensional state for
        implicit scheme Newton iteration
        """

        # TODO: add hyper-reduction

        # TODO: scaled_trial_basis should be calculated once
        scaled_trial_basis = self.trial_basis * self.norm_fac_prof_cons.ravel(order="C")[:, None]

        # Compute test basis
        test_basis = (res_jacob.toarray() / self.norm_fac_prof_cons.ravel(order="C")[:, None]) @ scaled_trial_basis

        # lhs and rhs of Newton iteration
        lhs = test_basis.T @ test_basis
        rhs = test_basis.T @ (res / self.norm_fac_prof_cons).ravel(order="C")

        # Linear solve
        dCode = np.linalg.solve(lhs, rhs)

        return dCode, lhs, rhs

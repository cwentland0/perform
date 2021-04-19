import numpy as np

from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearSPLSVTProj(LinearProjROM):
    """Class for projection-based ROM with linear decoder and SP-LSVT projection.
    
    Inherits from LinearProjROM. 

    Trial basis is assumed to represent the primitive variables. Allows implicit time integration only.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        trial_basis_scaled: 2D NumPy array of trial basis scaled by norm_fac_prof_prim. Precomputed for cost savings.
        hyper_reduc_operator: 2D NumPy array of gappy POD projection operator. Precomputed for cost savings.
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
        """Compute change in low-dimensional state for implicit scheme Newton iteration.

        This function computes the iterative change in the low-dimensional state for a given Newton iteration
        of an implicit time integration scheme. For SP-LSVT projection, this is given by

        W^T * W * d_code = W^T * res

        Where

        W = P^-1 * res_jacob * H * V_p

        Args:
            res_jacob:
                scipy.sparse.csr_matrix containing full-dimensional residual Jacobian with respect to the
                primitive variables.
            res: NumPy array of fully-discrete residual, already negated for Newton iteration.
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

        Returns:
            d_code:
                Solution of low-dimensional linear solve, representing the iterative change in
                the low-dimensional state.
            lhs: Left-hand side of low-dimensional linear solve.
            rhs: Right-hand side of low-dimensional linear solve.
        """

        # compute test basis
        test_basis = (res_jacob @ self.trial_basis_scaled) / self.norm_fac_prof_cons.ravel(order="C")[
            self.direct_samp_idxs_flat, None
        ]
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

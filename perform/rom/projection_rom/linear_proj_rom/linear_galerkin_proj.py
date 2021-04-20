import numpy as np

from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearGalerkinProj(LinearProjROM):
    """Class for projection-based ROM with linear decoder and Galerkin projection.

    Inherits from LinearProjROM.

    Trial basis is assumed to represent the conservative variables. Allows implicit and explicit time integration.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        trial_basis_scaled: 2D NumPy array of trial basis scaled by norm_fac_prof_cons. Precomputed for cost savings.
        hyper_reduc_operator: 2D NumPy array of gappy POD projection operator. Precomputed for cost savings.
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
        """Compute RHS projection operator.

        Called by ProjectionROM.calc_rhs_low_dim() to compute projection operator which is applied to RHS function
        for explicit time integration.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
        """

        if self.hyper_reduc:
            self.projector = self.hyper_reduc_operator
        else:
            self.projector = self.trial_basis.T

    def calc_d_code(self, res_jacob, res, sol_domain):
        """Compute change in low-dimensional state for implicit scheme Newton iteration.

        This function computes the iterative change in the low-dimensional state for a given Newton iteration
        of an implicit time integration scheme. For Galerkin projection, this is given by

        V^T * P^-1 * res_jacob * P * V * d_code = V^T * P^-1 * res

        Args:
            res_jacob:
                scipy.sparse.csr_matrix containing full-dimensional residual Jacobian with respect to
                the conservative variables.
            res: NumPy array of fully-discrete residual, already negated for Newton iteration.
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

        Returns:
            d_code:
                Solution of low-dimensional linear solve, representing the iterative change in
                the low-dimensional state.
            lhs: Left-hand side of low-dimensional linear solve.
            rhs: Right-hand side of low-dimensional linear solve.
        """

        lhs = (res_jacob @ self.trial_basis_scaled) / self.norm_fac_prof_cons.ravel(order="C")[
            self.direct_samp_idxs_flat, None
        ]
        res_scaled = (res / self.norm_fac_prof_cons[:, sol_domain.direct_samp_idxs]).ravel(order="C")

        # Project LHS and RHS onto low-dimensional space, solve linear system
        if self.hyper_reduc:
            lhs = self.hyper_reduc_operator @ lhs
            rhs = self.hyper_reduc_operator @ res_scaled
        else:
            lhs = self.trial_basis.T @ lhs
            rhs = self.trial_basis.T @ res_scaled

        d_code = np.linalg.solve(lhs, rhs)

        return d_code, lhs, rhs

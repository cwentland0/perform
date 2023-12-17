import numpy as np

from perform.constants import REAL_TYPE
from perform.rom.rom_method.projection_method.projection_method import ProjectionMethod


class LSPGProjection(ProjectionMethod):
    """Intrusive LSPG projection.

    LSPG projection is intrusive w/ numerical time integration, and targets the conservative variables
    """

    def __init__(self, sol_domain, rom_domain, solver):

        # check ROM input
        rom_dict = rom_domain.rom_dict

        # check that conservative scaling profiles are provided
        assert "cent_cons" in rom_dict
        assert "norm_fac_cons" in rom_dict
        assert "norm_sub_cons" in rom_dict
        rom_dict["cent_profs"] = rom_dict["cent_cons"]
        rom_dict["norm_fac_profs"] = rom_dict["norm_fac_cons"]
        rom_dict["norm_sub_profs"] = rom_dict["norm_sub_cons"]

        # check that variable mapping maps to conservative variables
        if "var_mapping" in rom_dict:
            assert rom_dict["var_mapping"] == "conservative"
        else:
            rom_dict["var_mapping"] = "conservative"

        # LSPG should not be using explicit time integration
        if sol_domain.time_integrator.time_type == "explicit":
            raise ValueError(
                "LSPG with an explicit time integrator deteriorates to Galerkin, please use"
                + " Galerkin or select an implicit time integrator."
            )

        # check that time stepper is not using dual time-stepping
        if sol_domain.time_integrator.dual_time:
            raise ValueError("LSPG is intended for conservative variable evolution, please set dual_time = False")

        super().__init__(sol_domain, rom_domain, solver)

    def calc_d_code(self, res_jacob, res, sol_domain, rom_domain):
        """Compute change in low-dimensional state for implicit scheme Newton iteration.

        This acts more as a utility function than a class method, since it takes in rom_domain and
        operates over all models. This is due to the coupled nature of the ROM equations.

        This function computes the iterative change in the low-dimensional state for a given Newton iteration
        of an implicit time integration scheme. For least-squares Petrov-Galerkin projection, this is given by

        W^T * W * d_code = W^T * res

        Where

        W = P^-1 * res_jacob * P * V

        Args:
            res_jacob:
                scipy.sparse.csr_matrix containing full-dimensional residual Jacobian with respect to the
                conservative variables.
            res: NumPy array of fully-discrete residual, already negated for Newton iteration.
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
            rom_domain: RomDomain within which this RomModel is contained.

        Returns:
            lhs: Left-hand side of low-dimensional linear solve.
            rhs: Right-hand side of low-dimensional linear solve.
        """

        # compute (scaled) concatenated decoder Jacobians
        decoder_jacob_concat, scaled_decoder_jacob_concat = self.assemble_concat_decoder_jacobs(sol_domain, rom_domain)

        # compute test basis
        if rom_domain.num_models == 1:
            test_basis = np.array(res_jacob @ scaled_decoder_jacob_concat)
        else:
            test_basis = (res_jacob @ scaled_decoder_jacob_concat).toarray()

        # scaling
        res_scaled = np.zeros(test_basis.shape[0], dtype=REAL_TYPE)
        for model in rom_domain.model_list:
            space_mapping = model.space_mapping
            for iter_idx, var_idx in enumerate(model.var_idxs):
                row_slice = np.s_[var_idx * sol_domain.mesh.num_cells : (var_idx + 1) * sol_domain.mesh.num_cells]
                res_scaled[row_slice] = (
                    res[var_idx, :] / space_mapping.norm_fac_prof[iter_idx, sol_domain.direct_samp_idxs]
                )
                test_basis[row_slice, :] /= space_mapping.norm_fac_prof[iter_idx, self.direct_samp_idxs_flat, None]

        if self.hyper_reduc:
            raise ValueError("Hyper-reduction not fixed yet")
            res_scaled = self.hyper_reduc_operator @ res_scaled
            test_basis = self.hyper_reduc_operator @ test_basis

        # LHS and RHS of Newton iteration
        lhs = test_basis.T @ test_basis
        rhs = test_basis.T @ res_scaled

        return lhs, rhs

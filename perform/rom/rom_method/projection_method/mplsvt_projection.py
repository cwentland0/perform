import os

import numpy as np

from perform.constants import REAL_TYPE
from perform.rom.rom_method.projection_method.projection_method import ProjectionMethod


class MPLSVTProjection(ProjectionMethod):
    """Intrusive MP-LSVT projection.

    MP-LSVT projection is intrusive w/ numerical time integration, and targets the primitive variables
    """

    # TODO: MP-LSVT can technically target an arbitrary, complete set of variables
    # Need a way to check that proper residual (Jacobian) routines exist for a given mapping

    def __init__(self, sol_domain, rom_domain):

        # check ROM input
        rom_dict = rom_domain.rom_dict

        # check that primitive scaling profiles are provided
        assert "cent_prim" in rom_dict
        assert "norm_fac_prim" in rom_dict
        assert "norm_sub_prim" in rom_dict
        rom_dict["cent_profs"] = rom_dict["cent_prim"]
        rom_dict["norm_fac_profs"] = rom_dict["norm_fac_prim"]
        rom_dict["norm_sub_profs"] = rom_dict["norm_sub_prim"]

        # get conservative divisive scaling profiles
        assert "norm_fac_cons" in rom_dict
        assert len(rom_dict["norm_fac_cons"]) == rom_domain.num_models
        self.norm_fac_profs_cons = rom_dict["norm_fac_cons"]

        # check that variable mapping maps to conservative variables
        if "var_mapping" in rom_dict:
            assert rom_dict["var_mapping"] == "primitive"
        else:
            rom_dict["var_mapping"] = "primitive"

        # MP-LSVT can use explicit time integration, hasn't been implemented
        if sol_domain.time_integrator.time_type == "explicit":
            raise ValueError("Explicit MP-LSVT not implemented yet")

        # check that time stepper is using dual time-stepping
        if not sol_domain.time_integrator.dual_time:
            raise ValueError(
                "MP-LSVT is intended for non-conservative variable evolution, please set dual_time = False"
            )

        super().__init__(sol_domain, rom_domain)

    def init_method(self, sol_domain, rom_domain):

        # load conservative scaling
        for model_idx, rom_model in enumerate(rom_domain.model_list):
            rom_model.space_mapping.norm_fac_prof_cons = rom_model.space_mapping.load_feature_scaling(
                os.path.join(rom_domain.model_dir, self.norm_fac_profs_cons[model_idx]), default="ones"
            )

        super().init_method(sol_domain, rom_domain)

    def calc_d_code(self, res_jacob, res, sol_domain, rom_domain):
        """Compute change in low-dimensional state for implicit scheme Newton iteration.

        This acts more as a utility function than a class method, since it takes in rom_domain and
        operates over all models. This is due to the coupled nature of the ROM equations.

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
            rom_domain: RomDomain within which this RomModel is contained.

        Returns:
            lhs: Left-hand side of low-dimensional linear solve.
            rhs: Right-hand side of low-dimensional linear solve.
        """

        # compute test basis
        if rom_domain.num_models == 1:
            test_basis = np.array(res_jacob @ self.trial_basis_scaled_concat)
        else:
            test_basis = (res_jacob @ self.trial_basis_scaled_concat).toarray()

        # scaling
        res_scaled = np.zeros(test_basis.shape[0], dtype=REAL_TYPE)
        for model in rom_domain.model_list:
            space_mapping = model.space_mapping
            for iter_idx, var_idx in enumerate(model.var_idxs):
                row_slice = np.s_[var_idx * sol_domain.mesh.num_cells : (var_idx + 1) * sol_domain.mesh.num_cells]
                res_scaled[row_slice] = (
                    res[var_idx, :] / space_mapping.norm_fac_prof_cons[iter_idx, sol_domain.direct_samp_idxs]
                )
                test_basis[row_slice, :] /= space_mapping.norm_fac_prof_cons[iter_idx, self.direct_samp_idxs_flat, None]

        if self.hyper_reduc:
            res_scaled = self.hyper_reduc_operator @ res_scaled
            test_basis = self.hyper_reduc_operator @ test_basis

        # LHS and RHS of Newton iteration
        lhs = test_basis.T @ test_basis
        rhs = test_basis.T @ res_scaled

        return lhs, rhs

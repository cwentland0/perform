import numpy as np
from scipy.sparse import csr_matrix

from perform.constants import REAL_TYPE
from perform.input_funcs import catch_input
from perform.rom.rom_method.rom_method import RomMethod


class ProjectionMethod(RomMethod):
    """Base class for projection-based intrusive ROM methods."""

    def __init__(self, sol_domain, rom_domain):

        rom_dict = rom_domain.rom_dict
        assert rom_dict["space_mapping"] in ["linear", "autoencoder"]

        if rom_dict["space_mapping"] == "autoencoder":
            raise ValueError("code has not been fixed to accommodate autoencoder models")

        self.is_intrusive = True

        # all projection-based ROMs use numerical time integration
        rom_dict["time_stepper"] = "numerical"

        super().__init__(sol_domain, rom_domain)

        # Set up hyper-reduction, if necessary
        self.hyper_reduc = catch_input(rom_dict, "hyper_reduc", False)
        if self.hyper_reduc:
            rom_domain.load_hyper_reduc(sol_domain)

        # load and check gappy POD basis
        if self.hyper_reduc:
            raise ValueError("This is broken")
            hyper_reduc_basis = np.load(rom_domain.hyper_reduc_files[self.model_idx])

            assert hyper_reduc_basis.ndim == 3, "Hyper-reduction basis must have three axes"
            assert hyper_reduc_basis.shape[:2] == (
                sol_domain.gas_model.num_eqs,
                sol_domain.mesh.num_cells,
            ), "Hyper reduction basis must have shape [num_eqs, num_cells, numHRModes]"

            self.hyper_reduc_dim = rom_domain.hyper_reduc_dims[self.model_idx]
            hyper_reduc_basis = hyper_reduc_basis[:, :, : self.hyper_reduc_dim]
            self.hyper_reduc_basis = np.reshape(hyper_reduc_basis, (-1, self.hyper_reduc_dim), order="C")

            # indices for sampling flattened hyper_reduc_basis
            self.direct_samp_idxs_flat = np.zeros(rom_domain.num_samp_cells * self.num_vars, dtype=np.int32)
            for var_num in range(self.num_vars):
                idx1 = var_num * rom_domain.num_samp_cells
                idx2 = (var_num + 1) * rom_domain.num_samp_cells
                self.direct_samp_idxs_flat[idx1:idx2] = (
                    rom_domain.direct_samp_idxs + var_num * sol_domain.mesh.num_cells
                )

        else:
            self.direct_samp_idxs_flat = np.s_[:]

        # This was from Galerkin projection
        # if self.hyper_reduc:
        #     # procompute hyper-reduction projector, V^T * U * [S^T * U]^+
        #     self.hyper_reduc_operator = (
        #         self.trial_basis.T
        #         @ self.hyper_reduc_basis
        #         @ np.linalg.pinv(self.hyper_reduc_basis[self.direct_samp_idxs_flat, :])
        #     )

        # This was from LSPG projection
        # Precompute hyper-reduction projector, [S^T * U]^+
        # TODO: may want to have an option to compute U * [S^T * U]^+
        #   The matrix can be big, but it's necessary for conservative LSPG, different least-square problem
        # if self.hyper_reduc:
        #     self.hyper_reduc_operator = np.linalg.pinv(self.hyper_reduc_basis[self.direct_samp_idxs_flat, :])

    def init_method(self, sol_domain, rom_domain):
        """Utility function for generating concatenated trial bases for implicit solve.

        Operates over all models in rom_domain.
        """

        if (rom_domain.rom_dict["space_mapping"] == "linear") and (
            rom_domain.time_stepper.time_integrator.time_type == "implicit"
        ):
            if rom_domain.num_models == 1:
                self.trial_basis_concat = rom_domain.model_list[0].space_mapping.trial_basis.copy()
                self.trial_basis_scaled_concat = rom_domain.model_list[0].space_mapping.trial_basis_scaled.copy()
            else:
                num_cells = sol_domain.mesh.num_cells
                trial_basis_concat = np.zeros(
                    (num_cells * sol_domain.gas_model.num_eqs, rom_domain.latent_dim_total), dtype=REAL_TYPE
                )
                trial_basis_scaled_concat = trial_basis_concat.copy()
                latent_dim_idx = 0
                for model in rom_domain.model_list:
                    col_slice = np.s_[latent_dim_idx : latent_dim_idx + model.latent_dim]
                    for iter_idx, var_idx in enumerate(model.var_idxs):
                        row_slice_basis = np.s_[iter_idx * num_cells : (iter_idx + 1) * num_cells]
                        row_slice_concat = np.s_[var_idx * num_cells : (var_idx + 1) * num_cells]
                        trial_basis_concat[row_slice_concat, col_slice] = model.trial_basis[row_slice_basis, :]
                        trial_basis_scaled_concat[row_slice_concat, col_slice] = model.trial_basis_scaled[
                            row_slice_basis, :
                        ]
                    latent_dim_idx += model.latent_dim
                self.trial_basis_concat = np.array(trial_basis_concat)
                self.trial_basis_scaled_concat = csr_matrix(trial_basis_scaled_concat)

    def project_to_low_dim(self, projector, full_dim_arr, transpose=False):
        """Project given full-dimensional vector onto low-dimensional space via given projector.

        Assumes that full_dim_arr is either 1D array or is in [num_vars, num_cells] order.
        Further assumes that projector is already in [latent_dim, num_vars x num_cells] order.

        Args:
            projector: 2D NumPy array containing linear projector.
            full_dim_arr: NumPy array of full-dimensional vector to be projected.
            transpose: If True, transposes projector before projecting full_dim_arr.

        Returns:
            NumPy array of low-dimensional projection of full_dim_arr.
        """

        if full_dim_arr.ndim == 2:
            full_dim_vec = full_dim_arr.flatten(order="C")
        elif full_dim_arr.ndim == 1:
            full_dim_vec = full_dim_arr.copy()
        else:
            raise ValueError("full_dim_arr must be one- or two-dimensional")

        if transpose:
            low_dim_vec = projector.T @ full_dim_vec
        else:
            low_dim_vec = projector @ full_dim_vec

        return low_dim_vec

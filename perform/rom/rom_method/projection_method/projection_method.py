import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import pinv

from perform.constants import REAL_TYPE
from perform.input_funcs import catch_input
from perform.rom.rom_method.rom_method import RomMethod
import sys


class ProjectionMethod(RomMethod):
    """Base class for projection-based intrusive ROM methods."""

    def __init__(self, sol_domain, rom_domain, solver):

        rom_dict = rom_domain.rom_dict
        assert rom_dict["space_mapping"] in ["linear", "autoencoder"]

        self.is_intrusive = True

        # all projection-based ROMs use numerical time integration
        rom_dict["time_stepper"] = "numerical"

        super().__init__(sol_domain, rom_domain, solver)

        # Set up hyper-reduction, if necessary
        #self.hyper_reduc = catch_input(rom_dict, "hyper_reduc", False)

        # TODO: this should be merged with the adaptive rom hyper-reduction so that we don't need the second condition.
        if not rom_domain.adaptive_rom:
            if rom_domain.hyper_reduc:
                rom_domain.load_hyper_reduc(rom_domain, sol_domain) 

        # load and check gappy POD basis
        if rom_domain.hyper_reduc:
            # TODO: this won't work if we have more than one model (adaptive ROM is also not implemented for more than one model)
            model_idx = 0
            # Not sure what was the issue with hyper-reduction that is not fixed!
            #raise ValueError("Hyper-reduction has not been fixed")
            if isinstance(rom_domain.hyper_reduc_files[model_idx], np.ndarray):
                hyper_reduc_basis = rom_domain.hyper_reduc_files[model_idx]
            else:
                hyper_reduc_basis = np.load(rom_domain.hyper_reduc_files[model_idx])

            assert hyper_reduc_basis.ndim == 3, "Hyper-reduction basis must have three axes"
            assert hyper_reduc_basis.shape[:2] == (
                sol_domain.gas_model.num_eqs,
                sol_domain.mesh.num_cells,
            ), "Hyper reduction basis must have shape [num_eqs, num_cells, numHRModes]"

            self.hyper_reduc_dim = rom_domain.hyper_reduc_dims[model_idx]
            hyper_reduc_basis = hyper_reduc_basis[:, :, : self.hyper_reduc_dim]
            self.hyper_reduc_basis = np.reshape(hyper_reduc_basis, (-1, self.hyper_reduc_dim), order="C")


            # indices for sampling flattened hyper_reduc_basis
            self.direct_samp_idxs_flat = np.zeros(rom_domain.num_samp_cells * rom_domain.num_vars, dtype=np.int32)
            for var_num in range(rom_domain.num_vars):
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
        if rom_domain.hyper_reduc:
            self.hyper_reduc_operator = np.linalg.pinv(self.hyper_reduc_basis[self.direct_samp_idxs_flat, :])

    def init_method(self, sol_domain, rom_domain):
        """Utility function for generating concatenated trial bases for implicit solve.

        Operates over all models in rom_domain.
        """

        if (rom_domain.rom_dict["space_mapping"] == "linear") and (
            rom_domain.time_stepper.time_integrator.time_type == "implicit"
        ):

            self.trial_basis_concat = self.concat_mapping(sol_domain, rom_domain, "trial_basis")
            self.trial_basis_scaled_concat = self.concat_mapping(
                sol_domain, rom_domain, "trial_basis_scaled", to_sparse=True
            )

    def concat_mapping(self, sol_domain, rom_domain, map_attr_str, to_sparse=False):

        if rom_domain.num_models == 1:
            mapping_concat = getattr(rom_domain.model_list[0].space_mapping, map_attr_str).copy()
        else:
            num_cells = sol_domain.mesh.num_cells
            mapping_concat = np.zeros(
                (num_cells * sol_domain.gas_model.num_eqs, rom_domain.latent_dim_total), dtype=REAL_TYPE
            )
            latent_dim_idx = 0
            for model in rom_domain.model_list:
                col_slice = np.s_[latent_dim_idx : latent_dim_idx + model.latent_dim]
                mapping = getattr(model, map_attr_str)
                for iter_idx, var_idx in enumerate(model.var_idxs):
                    row_slice_basis = np.s_[iter_idx * num_cells : (iter_idx + 1) * num_cells]
                    row_slice_concat = np.s_[var_idx * num_cells : (var_idx + 1) * num_cells]
                    mapping_concat[row_slice_concat, col_slice] = mapping[row_slice_basis, :]
                latent_dim_idx += model.latent_dim

            if to_sparse:
                mapping_concat = csr_matrix(mapping)
            else:
                mapping_concat = mapping

        return mapping_concat

    def assemble_concat_decoder_jacobs(self, sol_domain, rom_domain):
        """Utility function for computing concatenated and (un)scaled decoder Jacobian

        Required for implicit solve, due to coupled natured of system.
        For linear bases, this has already been precomputed.
        TODO: for adaptive bases, this will change
        """

        # if the space mapping is fixed linear, this has already been precomputed
        if rom_domain.rom_dict["space_mapping"] == "linear":
            decoder_jacob_concat = self.trial_basis_concat
            scaled_decoder_jacob_concat = self.trial_basis_scaled_concat

        else:

            # calculate all decoder Jacobians
            for model in rom_domain.model_list:
                space_mapping = model.space_mapping
                space_mapping.decoder_jacob = space_mapping.calc_decoder_jacob(model.code)
                space_mapping.scaled_decoder_jacob = (
                    space_mapping.decoder_jacob * space_mapping.norm_fac_prof.ravel(order="C")[:, None]
                )

            # concatenate decoder Jacobians
            decoder_jacob_concat = self.concat_mapping(sol_domain, rom_domain, "decoder_jacob", to_sparse=False)
            scaled_decoder_jacob_concat = self.concat_mapping(
                sol_domain, rom_domain, "scaled_decoder_jacob", to_sparse=True
            )

        return decoder_jacob_concat, scaled_decoder_jacob_concat

    def calc_concat_jacob_pinv(self, rom_domain, jacob):

        # TODO: may not always be orthonormal
        if rom_domain.rom_dict["space_mapping"] == "linear":
            jacob_pinv = jacob.T
        else:
            jacob_pinv = pinv(jacob)
        return jacob_pinv

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

    # def load_hyper_reduc(self, rom_domain, sol_domain, samp_idx = [], hyper_reduc_basis = [], hyper_reduc_dims = []):
    #     """Loads direct sampling indices and determines cell indices for hyper-reduction array slicing.

    #     Numerous array slicing indices are required for various operations in efficiently computing
    #     the non-linear RHS term, such as calculating fluxes, gradients, source terms, etc. as well as for computing
    #     the RHS Jacobian if required. These slicing arrays are first generated here based on the initial sampling
    #     indices, but may later be updated during sampling adaptation.

    #     Todos:
    #         Many of these operations should be moved to their own separate functions when
    #         recomputing sampling for adaptive sampling.

    #     Args:
    #         sol_domain: SolutionDomain with which this RomDomain is associated.
    #     """

    #     #raise ValueError("Hyper-reduction not fixed yet")

    #     # # TODO: add some explanations for what each index array accomplishes

    #     if not isinstance(samp_idx, np.ndarray):
    #         # load and check sample points
    #         samp_file = catch_input(rom_domain.rom_dict, "samp_file", "")
    #         assert samp_file != "", "Must supply samp_file if performing hyper-reduction"
    #         samp_file = os.path.join(rom_domain.model_dir, samp_file)
    #         assert os.path.isfile(samp_file), "Could not find samp_file at " + samp_file

    #         # Indices of directly sampled cells, within sol_prim/cons
    #         # NOTE: assumed that sample indices are zero-indexed
    #         sol_domain.direct_samp_idxs = np.load(samp_file).flatten()
    #     else:
    #         sol_domain.direct_samp_idxs = samp_idx.flatten()

    #     sol_domain.direct_samp_idxs = (np.sort(sol_domain.direct_samp_idxs)).astype(np.int32)
    #     sol_domain.num_samp_cells = len(sol_domain.direct_samp_idxs)
    #     assert (
    #         sol_domain.num_samp_cells <= sol_domain.mesh.num_cells
    #     ), "Cannot supply more sampling points than cells in domain."
    #     assert np.amin(sol_domain.direct_samp_idxs) >= 0, "Sampling indices must be non-negative integers"
    #     assert (
    #         np.amax(sol_domain.direct_samp_idxs) < sol_domain.mesh.num_cells
    #     ), "Sampling indices must be less than the number of cells in the domain"
    #     assert (
    #         len(np.unique(sol_domain.direct_samp_idxs)) == sol_domain.num_samp_cells
    #     ), "Sampling indices must be unique"


    #     # Copy indices for ease of use
    #     rom_domain.num_samp_cells = sol_domain.num_samp_cells
    #     rom_domain.direct_samp_idxs = sol_domain.direct_samp_idxs

    #     # Paths to hyper-reduction files (unpacked later)
    #     rom_domain.hyper_reduc_files = [None] * rom_domain.num_models
    #     if hyper_reduc_basis == []:
    #         hyper_reduc_files = rom_domain.rom_dict["hyper_reduc_files"]
    #         assert len(hyper_reduc_files) == rom_domain.num_models, "Must provide hyper_reduc_files for each model"
    #         for model_idx in range(rom_domain.num_models):
    #             in_file = os.path.join(rom_domain.model_dir, hyper_reduc_files[model_idx])
    #             assert os.path.isfile(in_file), "Could not find hyper-reduction file at " + in_file
    #             rom_domain.hyper_reduc_files[model_idx] = in_file
    #     else:
    #         for model_idx in range(rom_domain.num_models):
    #             rom_domain.hyper_reduc_files[model_idx] = hyper_reduc_basis[model_idx]

    #     # Load hyper reduction dimensions and check validity
    #     if hyper_reduc_dims != []:
    #         rom_domain.hyper_reduc_dims = hyper_reduc_dims
    #     else:
    #         rom_domain.hyper_reduc_dims = catch_list(
    #           rom_domain.rom_dict,
    #           "hyper_reduc_dims",
    #           [0],
    #           len_highest=rom_domain.num_models
    #         )

    #     for i in rom_domain.hyper_reduc_dims:
    #         assert i > 0, "hyper_reduc_dims must contain positive integers"
    #     if rom_domain.num_models == 1:
    #         assert (
    #             len(rom_domain.hyper_reduc_dims) == 1
    #         ), "Must provide only one value of hyper_reduc_dims when num_models = 1"
    #         assert rom_domain.hyper_reduc_dims[0] > 0, "hyper_reduc_dims must contain positive integers"
    #     else:
    #         if len(rom_domain.hyper_reduc_dims) == rom_domain.num_models:
    #             pass
    #         elif len(rom_domain.hyper_reduc_dims) == 1:
    #             print("Only one value provided in hyper_reduc_dims, applying to all models")
    #             sleep(1.0)
    #             rom_domain.hyper_reduc_dims = [rom_domain.hyper_reduc_dims[0]] * rom_domain.num_models
    #         else:
    #             raise ValueError("Must provide either num_models or 1 entry in hyper_reduc_dims")

    #     self.compute_cellidx_hyper_reduc(sol_domain)
        
    # def compute_cellidx_hyper_reduc(self, sol_domain):
    #     # Moved part of load_hyper_reduc here so that this function can be called if DEIM interpolation points are adapted
        
    #     # Compute indices for inviscid flux calculations
    #     # NOTE: have to account for fact that boundary cells are prepended/appended
    #     # Indices of "left" cells for flux calcs, within sol_prim/cons_full
    #     sol_domain.flux_samp_left_idxs = np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
    #     sol_domain.flux_samp_left_idxs[0::2] = sol_domain.direct_samp_idxs
    #     sol_domain.flux_samp_left_idxs[1::2] = sol_domain.direct_samp_idxs + 1

    #     # Indices of "right" cells for flux calcs, within sol_prim/cons_full
    #     sol_domain.flux_samp_right_idxs = np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
    #     sol_domain.flux_samp_right_idxs[0::2] = sol_domain.direct_samp_idxs + 1
    #     sol_domain.flux_samp_right_idxs[1::2] = sol_domain.direct_samp_idxs + 2

    #     # Eliminate repeated indices
    #     sol_domain.flux_samp_left_idxs = np.unique(sol_domain.flux_samp_left_idxs)
    #     sol_domain.flux_samp_right_idxs = np.unique(sol_domain.flux_samp_right_idxs)
    #     sol_domain.num_flux_faces = len(sol_domain.flux_samp_left_idxs)

    #     # Indices of flux array which correspond to left face of cell and map to direct_samp_idxs
    #     sol_domain.flux_rhs_idxs = np.zeros(sol_domain.num_samp_cells, np.int32)
    #     for i in range(1, sol_domain.num_samp_cells):
    #         # if this cell is adjacent to previous sampled cell
    #         if sol_domain.direct_samp_idxs[i] == (sol_domain.direct_samp_idxs[i - 1] + 1):
    #             sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 1
    #         # otherwise
    #         else:
    #             sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 2

    #     # Compute indices for gradient calculations
    #     # NOTE: also need to account for prepended/appended boundary cells
    #     # TODO: generalize for higher-order schemes
    #     if sol_domain.space_order > 1:
    #         if sol_domain.space_order == 2:

    #             # Indices of cells for which gradients need to be calculated, within sol_prim/cons_full
    #             sol_domain.grad_idxs = np.concatenate(
    #                 (
    #                     sol_domain.direct_samp_idxs + 1,
    #                     sol_domain.direct_samp_idxs,
    #                     sol_domain.direct_samp_idxs + 2,
    #                 )
    #             )
    #             sol_domain.grad_idxs = np.unique(sol_domain.grad_idxs)

    #             # Exclude left neighbor of inlet, right neighbor of outlet
    #             if sol_domain.grad_idxs[0] == 0:
    #                 sol_domain.grad_idxs = sol_domain.grad_idxs[1:]

    #             if sol_domain.grad_idxs[-1] == (sol_domain.mesh.num_cells + 1):
    #                 sol_domain.grad_idxs = sol_domain.grad_idxs[:-1]

    #             sol_domain.num_grad_cells = len(sol_domain.grad_idxs)

    #             # Indices of gradient cells and their immediate neighbors, within sol_prim/cons_full
    #             sol_domain.grad_neigh_idxs = np.concatenate((sol_domain.grad_idxs - 1, sol_domain.grad_idxs + 1))
    #             sol_domain.grad_neigh_idxs = np.unique(sol_domain.grad_neigh_idxs)

    #             # Exclude left neighbor of inlet, right neighbor of outlet
    #             if sol_domain.grad_neigh_idxs[0] == -1:
    #                 sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[1:]

    #             if sol_domain.grad_neigh_idxs[-1] == (sol_domain.mesh.num_cells + 2):
    #                 sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[:-1]

    #             # Indices within gradient neighbor indices to extract gradient cells, excluding boundaries
    #             _, _, sol_domain.grad_neigh_extract = np.intersect1d(
    #                 sol_domain.grad_idxs,
    #                 sol_domain.grad_neigh_idxs,
    #                 return_indices=True,
    #             )

    #             # Indices of grad_idxs in flux_samp_left_idxs and flux_samp_right_idxs and vice versa
    #             _, sol_domain.grad_left_extract, sol_domain.flux_left_extract = np.intersect1d(
    #                 sol_domain.grad_idxs,
    #                 sol_domain.flux_samp_left_idxs,
    #                 return_indices=True,
    #             )

    #             # Indices of grad_idxs in flux_samp_right_idxs and flux_samp_right_idxs and vice versa
    #             _, sol_domain.grad_right_extract, sol_domain.flux_right_extract = np.intersect1d(
    #                 sol_domain.grad_idxs,
    #                 sol_domain.flux_samp_right_idxs,
    #                 return_indices=True,
    #             )

    #         else:
    #             raise ValueError("Sampling for higher-order schemes not implemented yet")

    #     # for Jacobian calculations
    #     if sol_domain.direct_samp_idxs[0] == 0:
    #         sol_domain.jacob_left_samp = sol_domain.flux_rhs_idxs[1:].copy()
    #     else:
    #         sol_domain.jacob_left_samp = sol_domain.flux_rhs_idxs.copy()

    #     if sol_domain.direct_samp_idxs[-1] == (sol_domain.sol_int.num_cells - 1):
    #         sol_domain.jacob_right_samp = sol_domain.flux_rhs_idxs[:-1].copy() + 1
    #     else:
    #         sol_domain.jacob_right_samp = sol_domain.flux_rhs_idxs.copy() + 1

    #     # re-initialize solution objects to proper size
    #     gas = sol_domain.gas_model
    #     ones_prof = np.ones((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
    #     sol_domain.sol_left = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)
    #     sol_domain.sol_right = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)

    #     if sol_domain.invisc_flux_name == "roe":
    #         ones_prof = np.ones((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
    #         sol_domain.sol_ave = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)

    #     # Redo CSR matrix indices for sparse Jacobian
    #     num_cells = sol_domain.mesh.num_cells
    #     num_samp_cells = sol_domain.num_samp_cells
    #     num_elements_center = gas.num_eqs ** 2 * num_samp_cells
    #     if sol_domain.direct_samp_idxs[0] == 0:
    #         num_elements_lower = gas.num_eqs ** 2 * (num_samp_cells - 1)
    #     else:
    #         num_elements_lower = num_elements_center
    #     if sol_domain.direct_samp_idxs[-1] == (num_cells - 1):
    #         num_elements_upper = gas.num_eqs ** 2 * (num_samp_cells - 1)
    #     else:
    #         num_elements_upper = num_elements_center
    #     sol_domain.sol_int.jacob_dim_first = gas.num_eqs * num_samp_cells
    #     sol_domain.sol_int.jacob_dim_second = gas.num_eqs * num_cells

    #     row_idxs_center = np.zeros(num_elements_center, dtype=np.int32)
    #     col_idxs_center = np.zeros(num_elements_center, dtype=np.int32)
    #     row_idxs_upper = np.zeros(num_elements_upper, dtype=np.int32)
    #     col_idxs_upper = np.zeros(num_elements_upper, dtype=np.int32)
    #     row_idxs_lower = np.zeros(num_elements_lower, dtype=np.int32)
    #     col_idxs_lower = np.zeros(num_elements_lower, dtype=np.int32)

    #     lin_idx_A = 0
    #     lin_idx_B = 0
    #     lin_idx_C = 0
    #     for i in range(gas.num_eqs):
    #         for j in range(gas.num_eqs):
    #             for k in range(num_samp_cells):

    #                 row_idxs_center[lin_idx_A] = i * num_samp_cells + k
    #                 col_idxs_center[lin_idx_A] = j * num_cells + sol_domain.direct_samp_idxs[k]
    #                 lin_idx_A += 1

    #                 if sol_domain.direct_samp_idxs[k] < (num_cells - 1):
    #                     row_idxs_upper[lin_idx_B] = i * num_samp_cells + k
    #                     col_idxs_upper[lin_idx_B] = j * num_cells + sol_domain.direct_samp_idxs[k] + 1
    #                     lin_idx_B += 1

    #                 if sol_domain.direct_samp_idxs[k] > 0:
    #                     row_idxs_lower[lin_idx_C] = i * num_samp_cells + k
    #                     col_idxs_lower[lin_idx_C] = j * num_cells + sol_domain.direct_samp_idxs[k] - 1
    #                     lin_idx_C += 1

    #     sol_domain.sol_int.jacob_row_idxs = np.concatenate((row_idxs_center, row_idxs_lower, row_idxs_upper))
    #     sol_domain.sol_int.jacob_col_idxs = np.concatenate((col_idxs_center, col_idxs_lower, col_idxs_upper))

    #     # Gamma inverse indices
    #     # TODO: once the conservative Jacobians get implemented, this is unnecessary, remove and clean
    #     if sol_domain.time_integrator.dual_time:
    #         sol_domain.gamma_idxs = sol_domain.direct_samp_idxs
    #     else:
    #         sol_domain.gamma_idxs = np.concatenate(
    #             (sol_domain.direct_samp_idxs, sol_domain.direct_samp_idxs + 1, sol_domain.direct_samp_idxs - 1)
    #         )
    #         sol_domain.gamma_idxs = np.unique(sol_domain.gamma_idxs)
    #         if sol_domain.gamma_idxs[0] == -1:
    #             sol_domain.gamma_idxs = sol_domain.gamma_idxs[1:]
    #         if sol_domain.gamma_idxs[-1] == sol_domain.mesh.num_cells:
    #             sol_domain.gamma_idxs = sol_domain.gamma_idxs[:-1]

    #     _, sol_domain.gamma_idxs_center, _ = np.intersect1d(
    #         sol_domain.gamma_idxs,
    #         sol_domain.direct_samp_idxs,
    #         return_indices=True,
    #     )

    #     _, sol_domain.gamma_idxs_left, _ = np.intersect1d(
    #         sol_domain.gamma_idxs,
    #         sol_domain.direct_samp_idxs - 1,
    #         return_indices=True,
    #     )

    #     _, sol_domain.gamma_idxs_right, _ = np.intersect1d(
    #         sol_domain.gamma_idxs,
    #         sol_domain.direct_samp_idxs + 1,
    #         return_indices=True,
    #     )

    def flatten_deim_idxs(self, rom_domain, sol_domain):
        
        self.direct_samp_idxs_flat = np.zeros(rom_domain.num_samp_cells * rom_domain.num_vars, dtype=np.int32)
        for var_num in range(rom_domain.num_vars):
            idx1 = var_num * rom_domain.num_samp_cells
            idx2 = (var_num + 1) * rom_domain.num_samp_cells
            self.direct_samp_idxs_flat[idx1:idx2] = (
                rom_domain.direct_samp_idxs + var_num * sol_domain.mesh.num_cells
            )
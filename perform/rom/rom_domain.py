import os
from time import sleep

import numpy as np

from perform.constants import REAL_TYPE
from perform.input_funcs import read_input_file, catch_list, catch_input
from perform.solution.solution_phys import SolutionPhys
from perform.rom import get_rom_method, get_variable_mapping, get_time_stepper
from perform.rom.rom_model import RomModel

# TODO: initializing latent code from files (removed from previous version)


class RomDomain:
    """Container class for all ROM models to be applied within a given SolutionDomain.

    The concept of a ROM solution being composed of multiple ROM models derives from the concept of
    "vector" vs. "scalar" ROMs initially referenced by Yuxiang Zhou in their 2010 Master's thesis.
    The "vector" concept follows the most common scenario in which a ROM provides a single mapping from a
    low-dimensional state to the complete physical state vector. The "scalar" concept is less common, whereby
    several ROM models map to separate subsets of the physical state variables (e.g. one model maps to density
    and energy, while another maps to momentum and density-weighted species mass fraction). Thus, some container
    for separate models is necessary.

    When running a ROM simulation, perform.driver.main() will generate a RomDomain for each SolutionDomain for which
    a ROM simulation is requested. The RomDomain will handle reading in the input parameters from the ROM parameter
    input file, checking the validity of these parameters, and initializing all requested RomModel's.

    During simulation runtime, RomDomain is responsible for executing most of each RomModel's higher-level functions
    and executing accessory functions, e.g. filtering. Beyond this, member functions of RomDomain generally handle
    operations that apply to the entire ROM solution, e.g. time integration, calculating residual norms, etc.

    Args:
        sol_domain: SolutionDomain with which this RomDomain is associated.
        solver: SystemSolver containing global simulation parameters.

    Attributes:
        rom_dict: Dictionary of parameters read from ROM parameter input file.
        rom_method: String of ROM method to be applied (e.g. "LinearGalerkinProj").
        num_models: Number of separate models encapsulated by RomDomain.
        latent_dims: List of latent variable dimensions for each RomModel.
        model_var_idxs: List of list of zero-indexed indices indicating which state variables each RomModel maps to.
        model_dir: String path to directory containing all files required to execute each RomModel.
        model_files:
            list of strings of file names associated with each RomModel's primary data structure
            (e.g. a linear model's trail basis), relative to model_dir.
        cent_ic: Boolean flag of whether the initial condition file should be used to center the solution profile.
        norm_sub_cons_in:
            list of strings of file names associated with each RomModel's conservative variable subtractive
            normalization profile, if needed, relative to model_dir.
        norm_fac_cons_in:
            list of strings of file names associated with each RomModel's conservative variable divisive
            normalization profile, if needed, relative to model_dir.
        cent_cons_in:
            list of strings of file names associated with each RomModel's conservative variable centering profile,
            if needed, relative to model_dir.
        norm_sub_prim_in:
            list of strings of file names associated with each RomModel's primitive variable subtractive
            normalization profile, if needed, relative to model_dir.
        norm_fac_prim_in:
            list of strings of file names associated with each RomModel's primitive variable divisive
            normalization profile, if needed, relative to model_dir.
        cent_prim_in:
            list of strings of file names associated with each RomModel's primitive variable centering profile
            if needed, relative to model_dir.
        has_time_integrator: Boolean flag indicating whether a given rom_method requires numerical time integration.
        is_intrusive:
            Boolean flag indicating whether a given rom_method is intrusive,
            i.e. requires computation of the governing equations RHS and its Jacobian.
        target_cons: Boolean flag indicating whether a given rom_method maps to the conservative variables.
        target_prim: Boolean flag indicating whether a given rom_method maps to the primitive variables.
        has_cons_norm:
            Boolean flag indicating whether a given rom_method requires conservative variable normalization profiles.
        has_cons_cent:
            Boolean flag indicating whether a given rom_method requires conservative variable centering profiles.
        has_prim_norm:
            Boolean flag indicating whether a given rom_method requires primitive variable normalization profiles.
        has_prim_cent:
            Boolean flag indicating whether a given rom_method requires primitive variable centering profiles.
        hyper_reduc: Boolean flag indicating whether hyper-reduction is to be used for an intrusive rom_method.
        model_list: list containing num_models RomModel objects associated with this RomDomain.
        code_init_files:
            list of strings of file names associated with low-dimensional state initialization profiles
            for each RomModel.
    """

    def __init__(self, sol_domain, solver):

        rom_dict = read_input_file(solver.rom_inputs)
        self.rom_dict = rom_dict

        # Load and check latent dimensions
        self.num_models = int(rom_dict["num_models"])
        self.latent_dims = catch_list(rom_dict, "latent_dims", [0], len_highest=self.num_models)
        self.latent_dim_total = 0
        for latent_dim in self.latent_dims:
            assert latent_dim > 0, "latent_dims must contain positive integers"
            self.latent_dim_total += latent_dim

        if self.num_models == 1:
            assert len(self.latent_dims) == 1, "Must provide only one value of latent_dims when num_models = 1"
            assert self.latent_dims[0] > 0, "latent_dims must contain positive integers"
        else:
            if len(self.latent_dims) == self.num_models:
                pass
            elif len(self.latent_dims) == 1:
                print("Only one value provided in latent_dims, applying to all models")
                sleep(1.0)
                self.latent_dims = [self.latent_dims[0]] * self.num_models
            else:
                raise ValueError("Must provide either num_models or 1 entry in latent_dims")

        # Load and check model_var_idxs
        self.model_var_idxs = catch_list(rom_dict, "model_var_idxs", [[-1]], len_highest=self.num_models)
        for model_idx in range(self.num_models):
            assert self.model_var_idxs[model_idx][0] != -1, "model_var_idxs input incorrectly, probably too few lists"
        assert len(self.model_var_idxs) == self.num_models, "Must specify model_var_idxs for every model"

        # Initialize RomMethod, RomVariableMapping, and RomTimeStepper
        self.rom_method = get_rom_method(rom_dict["rom_method"], sol_domain, self)
        self.var_mapping = get_variable_mapping(rom_dict["var_mapping"], sol_domain, self)
        self.time_stepper = get_time_stepper(rom_dict["time_stepper"], sol_domain, self)

        # Check model base directory
        self.model_dir = str(rom_dict["model_dir"])
        assert os.path.isdir(self.model_dir), "Could not find model_dir at " + self.model_dir

        # Check latent code init files
        self.code_init_files = catch_list(rom_dict, "code_init_files", [""])
        if (len(self.code_init_files) != 1) or (self.code_init_files[0] != ""):
            assert len(self.code_init_files) == self.num_models, (
                "If initializing any ROM model from a file, must provide list entries for every model. "
                + "If you don't wish to initialize from file for a model, input an empty string in the list entry."
            )
        else:
            self.code_init_files = [""] * self.num_models

        # Initialize RomModels (and associated RomSpaceMappings)
        self.model_list = [None] * self.num_models
        for model_idx in range(self.num_models):
            self.model_list[model_idx] = RomModel(model_idx, sol_domain, self)

        # Additional initializations specific to RomTimeStepper and RomMethod
        self.time_stepper.init_state(sol_domain, self)
        self.rom_method.init_method(sol_domain, self)

    def advance_iter(self, sol_domain, solver):
        """Advance low-dimensional state and full solution forward one physical time iteration.

        For non-intrusive ROMs without a time integrator, simply advances the solution one step.

        For intrusive and non-intrusive ROMs with a time integrator, begins numerical time integration
        and steps through sub-iterations.

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """

        print("Iteration " + str(solver.iter))
        self.time_stepper.advance_iter(sol_domain, solver, self)

        sol_domain.sol_int.update_sol_hist()
        self.update_code_hist()

    def update_code_hist(self):
        """Update low-dimensional state history after physical time step."""

        for model in self.model_list:

            model.code_hist[1:] = model.code_hist[:-1]
            model.code_hist[0] = model.code.copy()

    def load_hyper_reduc(self, sol_domain):
        """Loads direct sampling indices and determines cell indices for hyper-reduction array slicing.

        Numerous array slicing indices are required for various operations in efficiently computing
        the non-linear RHS term, such as calculating fluxes, gradients, source terms, etc. as well as for computing
        the RHS Jacobian if required. These slicing arrays are first generated here based on the initial sampling
        indices, but may later be updated during sampling adaptation.

        Todos:
            Many of these operations should be moved to their own separate functions when
            recomputing sampling for adaptive sampling.

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
        """

        # TODO: add some explanations for what each index array accomplishes

        # load and check sample points
        samp_file = catch_input(self.rom_dict, "samp_file", "")
        assert samp_file != "", "Must supply samp_file if performing hyper-reduction"
        samp_file = os.path.join(self.model_dir, samp_file)
        assert os.path.isfile(samp_file), "Could not find samp_file at " + samp_file

        # Indices of directly sampled cells, within sol_prim/cons
        # NOTE: assumed that sample indices are zero-indexed
        sol_domain.direct_samp_idxs = np.load(samp_file).flatten()
        sol_domain.direct_samp_idxs = (np.sort(sol_domain.direct_samp_idxs)).astype(np.int32)
        sol_domain.num_samp_cells = len(sol_domain.direct_samp_idxs)
        assert (
            sol_domain.num_samp_cells <= sol_domain.mesh.num_cells
        ), "Cannot supply more sampling points than cells in domain."
        assert np.amin(sol_domain.direct_samp_idxs) >= 0, "Sampling indices must be non-negative integers"
        assert (
            np.amax(sol_domain.direct_samp_idxs) < sol_domain.mesh.num_cells
        ), "Sampling indices must be less than the number of cells in the domain"
        assert (
            len(np.unique(sol_domain.direct_samp_idxs)) == sol_domain.num_samp_cells
        ), "Sampling indices must be unique"

        # Compute indices for inviscid flux calculations
        # NOTE: have to account for fact that boundary cells are prepended/appended
        # Indices of "left" cells for flux calcs, within sol_prim/cons_full
        sol_domain.flux_samp_left_idxs = np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
        sol_domain.flux_samp_left_idxs[0::2] = sol_domain.direct_samp_idxs
        sol_domain.flux_samp_left_idxs[1::2] = sol_domain.direct_samp_idxs + 1

        # Indices of "right" cells for flux calcs, within sol_prim/cons_full
        sol_domain.flux_samp_right_idxs = np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
        sol_domain.flux_samp_right_idxs[0::2] = sol_domain.direct_samp_idxs + 1
        sol_domain.flux_samp_right_idxs[1::2] = sol_domain.direct_samp_idxs + 2

        # Eliminate repeated indices
        sol_domain.flux_samp_left_idxs = np.unique(sol_domain.flux_samp_left_idxs)
        sol_domain.flux_samp_right_idxs = np.unique(sol_domain.flux_samp_right_idxs)
        sol_domain.num_flux_faces = len(sol_domain.flux_samp_left_idxs)

        # Indices of flux array which correspond to left face of cell and map to direct_samp_idxs
        sol_domain.flux_rhs_idxs = np.zeros(sol_domain.num_samp_cells, np.int32)
        for i in range(1, sol_domain.num_samp_cells):
            # if this cell is adjacent to previous sampled cell
            if sol_domain.direct_samp_idxs[i] == (sol_domain.direct_samp_idxs[i - 1] + 1):
                sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 1
            # otherwise
            else:
                sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 2

        # Compute indices for gradient calculations
        # NOTE: also need to account for prepended/appended boundary cells
        # TODO: generalize for higher-order schemes
        if sol_domain.space_order > 1:
            if sol_domain.space_order == 2:

                # Indices of cells for which gradients need to be calculated, within sol_prim/cons_full
                sol_domain.grad_idxs = np.concatenate(
                    (
                        sol_domain.direct_samp_idxs + 1,
                        sol_domain.direct_samp_idxs,
                        sol_domain.direct_samp_idxs + 2,
                    )
                )
                sol_domain.grad_idxs = np.unique(sol_domain.grad_idxs)

                # Exclude left neighbor of inlet, right neighbor of outlet
                if sol_domain.grad_idxs[0] == 0:
                    sol_domain.grad_idxs = sol_domain.grad_idxs[1:]

                if sol_domain.grad_idxs[-1] == (sol_domain.mesh.num_cells + 1):
                    sol_domain.grad_idxs = sol_domain.grad_idxs[:-1]

                sol_domain.num_grad_cells = len(sol_domain.grad_idxs)

                # Indices of gradient cells and their immediate neighbors, within sol_prim/cons_full
                sol_domain.grad_neigh_idxs = np.concatenate((sol_domain.grad_idxs - 1, sol_domain.grad_idxs + 1))
                sol_domain.grad_neigh_idxs = np.unique(sol_domain.grad_neigh_idxs)

                # Exclude left neighbor of inlet, right neighbor of outlet
                if sol_domain.grad_neigh_idxs[0] == -1:
                    sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[1:]

                if sol_domain.grad_neigh_idxs[-1] == (sol_domain.mesh.num_cells + 2):
                    sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[:-1]

                # Indices within gradient neighbor indices to extract gradient cells, excluding boundaries
                _, _, sol_domain.grad_neigh_extract = np.intersect1d(
                    sol_domain.grad_idxs,
                    sol_domain.grad_neigh_idxs,
                    return_indices=True,
                )

                # Indices of grad_idxs in flux_samp_left_idxs and flux_samp_right_idxs and vice versa
                _, sol_domain.grad_left_extract, sol_domain.flux_left_extract = np.intersect1d(
                    sol_domain.grad_idxs,
                    sol_domain.flux_samp_left_idxs,
                    return_indices=True,
                )

                # Indices of grad_idxs in flux_samp_right_idxs and flux_samp_right_idxs and vice versa
                _, sol_domain.grad_right_extract, sol_domain.flux_right_extract = np.intersect1d(
                    sol_domain.grad_idxs,
                    sol_domain.flux_samp_right_idxs,
                    return_indices=True,
                )

            else:
                raise ValueError("Sampling for higher-order schemes not implemented yet")

        # for Jacobian calculations
        if sol_domain.direct_samp_idxs[0] == 0:
            sol_domain.jacob_left_samp = sol_domain.flux_rhs_idxs[1:].copy()
        else:
            sol_domain.jacob_left_samp = sol_domain.flux_rhs_idxs.copy()

        if sol_domain.direct_samp_idxs[-1] == (sol_domain.sol_int.num_cells - 1):
            sol_domain.jacob_right_samp = sol_domain.flux_rhs_idxs[:-1].copy() + 1
        else:
            sol_domain.jacob_right_samp = sol_domain.flux_rhs_idxs.copy() + 1

        # re-initialize solution objects to proper size
        gas = sol_domain.gas_model
        ones_prof = np.ones((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
        sol_domain.sol_left = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)
        sol_domain.sol_right = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)

        if sol_domain.invisc_flux_name == "roe":
            ones_prof = np.ones((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
            sol_domain.sol_ave = SolutionPhys(gas, sol_domain.num_flux_faces, sol_prim_in=ones_prof)

        # Copy indices for ease of use
        self.num_samp_cells = sol_domain.num_samp_cells
        self.direct_samp_idxs = sol_domain.direct_samp_idxs

        # Paths to hyper-reduction files (unpacked later)
        hyper_reduc_files = self.rom_dict["hyper_reduc_files"]
        self.hyper_reduc_files = [None] * self.num_models
        assert len(hyper_reduc_files) == self.num_models, "Must provide hyper_reduc_files for each model"
        for model_idx in range(self.num_models):
            in_file = os.path.join(self.model_dir, hyper_reduc_files[model_idx])
            assert os.path.isfile(in_file), "Could not find hyper-reduction file at " + in_file
            self.hyper_reduc_files[model_idx] = in_file

        # Load hyper reduction dimensions and check validity
        self.hyper_reduc_dims = catch_list(self.rom_dict, "hyper_reduc_dims", [0], len_highest=self.num_models)

        for i in self.hyper_reduc_dims:
            assert i > 0, "hyper_reduc_dims must contain positive integers"
        if self.num_models == 1:
            assert (
                len(self.hyper_reduc_dims) == 1
            ), "Must provide only one value of hyper_reduc_dims when num_models = 1"
            assert self.hyper_reduc_dims[0] > 0, "hyper_reduc_dims must contain positive integers"
        else:
            if len(self.hyper_reduc_dims) == self.num_models:
                pass
            elif len(self.hyper_reduc_dims) == 1:
                print("Only one value provided in hyper_reduc_dims, applying to all models")
                sleep(1.0)
                self.hyper_reduc_dims = [self.hyper_reduc_dims[0]] * self.num_models
            else:
                raise ValueError("Must provide either num_models or 1 entry in hyper_reduc_dims")

        # Redo CSR matrix indices for sparse Jacobian
        num_cells = sol_domain.mesh.num_cells
        num_samp_cells = sol_domain.num_samp_cells
        num_elements_center = gas.num_eqs ** 2 * num_samp_cells
        if sol_domain.direct_samp_idxs[0] == 0:
            num_elements_lower = gas.num_eqs ** 2 * (num_samp_cells - 1)
        else:
            num_elements_lower = num_elements_center
        if sol_domain.direct_samp_idxs[-1] == (num_cells - 1):
            num_elements_upper = gas.num_eqs ** 2 * (num_samp_cells - 1)
        else:
            num_elements_upper = num_elements_center
        sol_domain.sol_int.jacob_dim_first = gas.num_eqs * num_samp_cells
        sol_domain.sol_int.jacob_dim_second = gas.num_eqs * num_cells

        row_idxs_center = np.zeros(num_elements_center, dtype=np.int32)
        col_idxs_center = np.zeros(num_elements_center, dtype=np.int32)
        row_idxs_upper = np.zeros(num_elements_upper, dtype=np.int32)
        col_idxs_upper = np.zeros(num_elements_upper, dtype=np.int32)
        row_idxs_lower = np.zeros(num_elements_lower, dtype=np.int32)
        col_idxs_lower = np.zeros(num_elements_lower, dtype=np.int32)

        lin_idx_A = 0
        lin_idx_B = 0
        lin_idx_C = 0
        for i in range(gas.num_eqs):
            for j in range(gas.num_eqs):
                for k in range(num_samp_cells):

                    row_idxs_center[lin_idx_A] = i * num_samp_cells + k
                    col_idxs_center[lin_idx_A] = j * num_cells + sol_domain.direct_samp_idxs[k]
                    lin_idx_A += 1

                    if sol_domain.direct_samp_idxs[k] < (num_cells - 1):
                        row_idxs_upper[lin_idx_B] = i * num_samp_cells + k
                        col_idxs_upper[lin_idx_B] = j * num_cells + sol_domain.direct_samp_idxs[k] + 1
                        lin_idx_B += 1

                    if sol_domain.direct_samp_idxs[k] > 0:
                        row_idxs_lower[lin_idx_C] = i * num_samp_cells + k
                        col_idxs_lower[lin_idx_C] = j * num_cells + sol_domain.direct_samp_idxs[k] - 1
                        lin_idx_C += 1

        sol_domain.sol_int.jacob_row_idxs = np.concatenate((row_idxs_center, row_idxs_lower, row_idxs_upper))
        sol_domain.sol_int.jacob_col_idxs = np.concatenate((col_idxs_center, col_idxs_lower, col_idxs_upper))

        # Gamma inverse indices
        # TODO: once the conservative Jacobians get implemented, this is unnecessary, remove and clean
        if sol_domain.time_integrator.dual_time:
            sol_domain.gamma_idxs = sol_domain.direct_samp_idxs
        else:
            sol_domain.gamma_idxs = np.concatenate(
                (sol_domain.direct_samp_idxs, sol_domain.direct_samp_idxs + 1, sol_domain.direct_samp_idxs - 1)
            )
            sol_domain.gamma_idxs = np.unique(sol_domain.gamma_idxs)
            if sol_domain.gamma_idxs[0] == -1:
                sol_domain.gamma_idxs = sol_domain.gamma_idxs[1:]
            if sol_domain.gamma_idxs[-1] == sol_domain.mesh.num_cells:
                sol_domain.gamma_idxs = sol_domain.gamma_idxs[:-1]

        _, sol_domain.gamma_idxs_center, _ = np.intersect1d(
            sol_domain.gamma_idxs,
            sol_domain.direct_samp_idxs,
            return_indices=True,
        )

        _, sol_domain.gamma_idxs_left, _ = np.intersect1d(
            sol_domain.gamma_idxs,
            sol_domain.direct_samp_idxs - 1,
            return_indices=True,
        )

        _, sol_domain.gamma_idxs_right, _ = np.intersect1d(
            sol_domain.gamma_idxs,
            sol_domain.direct_samp_idxs + 1,
            return_indices=True,
        )

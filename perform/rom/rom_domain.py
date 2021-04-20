import os
from time import sleep

import numpy as np

from perform.constants import REAL_TYPE
from perform.input_funcs import read_input_file, catch_list, catch_input
from perform.solution.solution_phys import SolutionPhys
from perform.time_integrator import get_time_integrator
from perform.rom import get_rom_model


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
        low_dim_init_files:
            list of strings of file names associated with low-dimensional state initialization profiles
            for each RomModel.
    """

    def __init__(self, sol_domain, solver):

        rom_dict = read_input_file(solver.rom_inputs)
        self.rom_dict = rom_dict

        # Load model parameters
        self.rom_method = str(rom_dict["rom_method"])
        self.num_models = int(rom_dict["num_models"])
        self.latent_dims = catch_list(rom_dict, "latent_dims", [0], len_highest=self.num_models)
        model_var_idxs = catch_list(rom_dict, "model_var_idxs", [[-1]], len_highest=self.num_models)

        # Check model parameters
        for i in self.latent_dims:
            assert i > 0, "latent_dims must contain positive integers"

        if self.num_models == 1:
            assert len(self.latent_dims) == 1, "Must provide only one value of latent_dims when num_models = 1"
            assert self.latent_dims[0] > 0, "latent_dims must contain positive integers"
        else:
            if len(self.latent_dims) == self.num_models:
                pass
            elif len(self.latent_dims) == 1:
                print("Only one value provided in latent_dims," + " applying to all models")
                sleep(1.0)
                self.latent_dims = [self.latent_dims[0]] * self.num_models
            else:
                raise ValueError("Must provide either num_models" + "or 1 entry in latent_dims")

        # Load and check model_var_idxs
        for model_idx in range(self.num_models):
            assert model_var_idxs[model_idx][0] != -1, "model_var_idxs input incorrectly, probably too few lists"
        assert len(model_var_idxs) == self.num_models, "Must specify model_var_idxs for every model"
        model_var_sum = 0
        for model_idx in range(self.num_models):
            model_var_sum += len(model_var_idxs[model_idx])
            for model_var_idx in model_var_idxs[model_idx]:
                assert model_var_idx >= 0, "model_var_idxs must be non-negative integers"
                assert (
                    model_var_idx < sol_domain.gas_model.num_eqs
                ), "model_var_idxs must less than the number of governing equations"
        assert model_var_sum == sol_domain.gas_model.num_eqs, (
            "Must specify as many model_var_idxs entries as governing equations ("
            + str(model_var_sum)
            + " != "
            + str(sol_domain.gas_model.num_eqs)
            + ")"
        )
        model_var_idxs_one_list = sum(model_var_idxs, [])
        assert len(model_var_idxs_one_list) == len(
            set(model_var_idxs_one_list)
        ), "All entries in model_var_idxs must be unique"
        self.model_var_idxs = model_var_idxs

        # Load and check model input locations
        self.model_dir = str(rom_dict["model_dir"])
        model_files = rom_dict["model_files"]
        self.model_files = [None] * self.num_models
        assert len(model_files) == self.num_models, "Must provide model_files for each model"
        for model_idx in range(self.num_models):
            in_file = os.path.join(self.model_dir, model_files[model_idx])
            assert os.path.isfile(in_file), "Could not find model file at " + in_file
            self.model_files[model_idx] = in_file

        # Load standardization profiles, if they are required
        self.cent_ic = catch_input(rom_dict, "cent_ic", False)
        self.norm_sub_cons_in = catch_list(rom_dict, "norm_sub_cons", [""])
        self.norm_fac_cons_in = catch_list(rom_dict, "norm_fac_cons", [""])
        self.cent_cons_in = catch_list(rom_dict, "cent_cons", [""])
        self.norm_sub_prim_in = catch_list(rom_dict, "norm_sub_prim", [""])
        self.norm_fac_prim_in = catch_list(rom_dict, "norm_fac_prim", [""])
        self.cent_prim_in = catch_list(rom_dict, "cent_prim", [""])

        self.set_model_flags()

        # Set up hyper-reduction, if necessary
        if self.is_intrusive:
            self.hyper_reduc = catch_input(rom_dict, "hyper_reduc", False)
            if self.hyper_reduc:
                self.load_hyper_reduc(sol_domain)

        # Get time integrator, if necessary
        # TODO: time_scheme should be specific to RomDomain, not the solver
        if self.has_time_integrator:
            self.time_integrator = get_time_integrator(solver.time_scheme, solver.param_dict)
        else:
            self.time_integrator = None  # TODO: this might be pointless

        # check init files
        self.low_dim_init_files = catch_list(rom_dict, "low_dim_init_files", [""])
        if (len(self.low_dim_init_files) != 1) or (self.low_dim_init_files[0] != ""):
            assert len(self.low_dim_init_files) == self.num_models, (
                "If initializing any ROM model from a file, must provide list entries for every model. "
                + "If you don't wish to initialize from file for a model, input an empty string "
                " in the list entry."
            )
        else:
            self.low_dim_init_files = [""] * self.num_models

        # Initialize
        self.model_list = [None] * self.num_models
        for model_idx in range(self.num_models):
            # Initialize model
            self.model_list[model_idx] = get_rom_model(model_idx, self, sol_domain)
            model = self.model_list[model_idx]

            # Initialize state
            init_file = self.low_dim_init_files[model_idx]
            if init_file != "":
                assert os.path.isfile(init_file), "Could not find ROM initialization file at " + init_file
                model.code = np.load(init_file)
                model.update_sol(sol_domain)
            else:
                model.init_from_sol(sol_domain)

            # Initialize code history
            model.code_hist = [model.code.copy()] * (self.time_integrator.time_order + 1)

        sol_domain.sol_int.update_state(from_cons=self.target_cons)

        # Overwrite history with initialized solution
        sol_domain.sol_int.sol_hist_cons = [sol_domain.sol_int.sol_cons.copy()] * (self.time_integrator.time_order + 1)
        sol_domain.sol_int.sol_hist_prim = [sol_domain.sol_int.sol_prim.copy()] * (self.time_integrator.time_order + 1)

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

        # Update model which does NOT require numerical time integration
        if not self.has_time_integrator:
            raise ValueError("Iteration advance for models without numerical time integration not yet implemented")

        # If method requires numerical time integration
        else:

            for self.time_integrator.subiter in range(self.time_integrator.subiter_max):

                self.advance_subiter(sol_domain, solver)

                if self.time_integrator.time_type == "implicit":
                    self.calc_code_res_norms(sol_domain, solver, self.time_integrator.subiter)

                    if sol_domain.sol_int.res_norm_l2 < self.time_integrator.res_tol:
                        break

        sol_domain.sol_int.update_sol_hist()
        self.update_code_hist()

    def advance_subiter(self, sol_domain, solver):
        """Advance low-dimensional state and full solution forward one subiteration of time integrator.

        For intrusive ROMs, computes RHS and RHS Jacobian (if necessary).

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """

        sol_int = sol_domain.sol_int
        res, res_jacob = None, None

        if self.is_intrusive:
            sol_domain.calc_rhs(solver)

        if self.time_integrator.time_type == "implicit":

            # Compute residual and residual Jacobian
            if self.is_intrusive:
                res = self.time_integrator.calc_residual(
                    sol_int.sol_hist_cons, sol_int.rhs, solver, samp_idxs=sol_domain.direct_samp_idxs
                )
                res_jacob = sol_domain.calc_res_jacob(solver)

            # Compute change in low-dimensional state
            for model_idx, model in enumerate(self.model_list):
                d_code, code_lhs, code_rhs = model.calc_d_code(res_jacob, res, sol_domain)
                model.code += d_code
                model.code_hist[0] = model.code.copy()
                model.update_sol(sol_domain)

                # Compute ROM residual for convergence measurement
                model.res = code_lhs @ d_code - code_rhs

            sol_int.update_state(from_cons=(not sol_domain.time_integrator.dual_time))
            sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
            sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

        else:

            for model_idx, model in enumerate(self.model_list):

                model.calc_rhs_low_dim(self, sol_domain)
                d_code = self.time_integrator.solve_sol_change(model.rhs_low_dim)
                model.code = model.code_hist[0] + d_code
                model.update_sol(sol_domain)

            sol_int.update_state(from_cons=True)

    def update_code_hist(self):
        """Update low-dimensional state history after physical time step."""

        for model in self.model_list:

            model.code_hist[1:] = model.code_hist[:-1]
            model.code_hist[0] = model.code.copy()

    def calc_code_res_norms(self, sol_domain, solver, subiter):
        """Calculate and print low-dimensional linear solve residual norms.

        Computes L2 and L1 norms of low-dimensional linear solve residuals for each RomModel,
        as computed in advance_subiter(). These are averaged across all RomModels and printed to the terminal,
        and are used in advance_iter() to determine whether the Newton's method iterative solve has
        converged sufficiently. If the norm is below numerical precision, it defaults to 1e-16.

        Note that terminal output is ORDER OF MAGNITUDE (i.e. 1e-X, where X is the order of magnitude).

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
            subiter: Current subiteration number within current time step's Newton's method iterative solve.
        """

        # Compute residual norm for each model
        norm_l2_sum = 0.0
        norm_l1_sum = 0.0
        for model in self.model_list:
            norm_l2, norm_l1 = model.calc_code_norms()
            norm_l2_sum += norm_l2
            norm_l1_sum += norm_l1

        # Average over all models
        norm_l2 = norm_l2_sum / self.num_models
        norm_l1 = norm_l1_sum / self.num_models

        # Norm is sometimes zero, just default to -16 I guess
        if norm_l2 == 0.0:
            norm_out_l2 = -16.0
        else:
            norm_out_l2 = np.log10(norm_l2)

        if norm_l1 == 0.0:
            norm_out_l1 = -16.0
        else:
            norm_out_l1 = np.log10(norm_l1)

        # Print to terminal
        out_string = (str(subiter + 1) + ":\tL2: %18.14f, \tL1: %18.14f") % (norm_out_l2, norm_out_l1,)
        print(out_string)

        sol_domain.sol_int.res_norm_l2 = norm_l2
        sol_domain.sol_int.resNormL1 = norm_l1
        sol_domain.sol_int.res_norm_hist[solver.iter - 1, :] = [norm_l2, norm_l1]

    def set_model_flags(self):
        """Set universal ROM method flags that dictate various execution behaviors.

        If a new RomModel is created, its flags should be set here.
        """

        self.has_time_integrator = False
        self.is_intrusive = False
        self.target_cons = False
        self.target_prim = False

        self.has_cons_norm = False
        self.has_cons_cent = False
        self.has_prim_norm = False
        self.has_prim_cent = False

        if self.rom_method == "linear_galerkin_proj":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "linear_lspg_proj":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "linear_splsvt_proj":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_prim = True
            self.has_cons_norm = True
            self.has_prim_norm = True
            self.has_prim_cent = True
        elif self.rom_method == "autoencoder_galerkin_proj_tfkeras":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "autoencoder_lspg_proj_tfkeras":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_cons = True
            self.has_cons_norm = True
            self.has_cons_cent = True
        elif self.rom_method == "autoencoder_splsvt_proj_tfkeras":
            self.has_time_integrator = True
            self.is_intrusive = True
            self.target_prim = True
            self.has_cons_norm = True
            self.has_prim_norm = True
            self.has_prim_cent = True
        else:
            raise ValueError("Invalid ROM method name: " + self.rom_method)

        # TODO: not strictly true for the non-intrusive models
        assert self.target_cons != self.target_prim, "Model must target either the primitive or conservative variables"

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
                    (sol_domain.direct_samp_idxs + 1, sol_domain.direct_samp_idxs, sol_domain.direct_samp_idxs + 2,)
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
                    sol_domain.grad_idxs, sol_domain.grad_neigh_idxs, return_indices=True,
                )

                # Indices of grad_idxs in flux_samp_left_idxs and flux_samp_right_idxs and vice versa
                _, sol_domain.grad_left_extract, sol_domain.flux_left_extract = np.intersect1d(
                    sol_domain.grad_idxs, sol_domain.flux_samp_left_idxs, return_indices=True,
                )

                # Indices of grad_idxs in flux_samp_right_idxs and flux_samp_right_idxs and vice versa
                _, sol_domain.grad_right_extract, sol_domain.flux_right_extract = np.intersect1d(
                    sol_domain.grad_idxs, sol_domain.flux_samp_right_idxs, return_indices=True,
                )

            else:
                raise ValueError("Sampling for higher-order schemes" + " not implemented yet")

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
            sol_domain.gamma_idxs, sol_domain.direct_samp_idxs, return_indices=True,
        )

        _, sol_domain.gamma_idxs_left, _ = np.intersect1d(
            sol_domain.gamma_idxs, sol_domain.direct_samp_idxs - 1, return_indices=True,
        )

        _, sol_domain.gamma_idxs_right, _ = np.intersect1d(
            sol_domain.gamma_idxs, sol_domain.direct_samp_idxs + 1, return_indices=True,
        )

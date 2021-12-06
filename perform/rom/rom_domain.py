import os
from time import sleep

from perform.input_funcs import catch_list, catch_input
from perform.rom import get_ml_library, get_rom_method, get_variable_mapping, get_time_stepper
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

        rom_dict = solver.rom_dict
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

        # get ML library, if requested
        self.ml_library = catch_input(rom_dict, "ml_library", "none")
        if self.ml_library != "none":
            self.mllib = get_ml_library(self.ml_library, self)

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

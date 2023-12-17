import os
from time import sleep
import copy
import numpy as np
from perform.constants import REAL_TYPE

from perform.input_funcs import catch_list, catch_input
from perform.rom import get_ml_library, get_rom_method, get_variable_mapping, get_time_stepper, gen_rom_basis, gen_deim_sampling
from perform.rom.rom_model import RomModel
from perform.solution.solution_phys import SolutionPhys
import sys

# TODO: initializing latent code from files (removed from previous version)

# TODO: it is better to transfer everything related to adaptive basis to projection_method.py
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
        self.adaptive_rom = catch_input(rom_dict, "adaptive_rom", False)
        self.hyper_reduc = catch_input(rom_dict, "hyper_reduc", False)

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

        # add rom dimension to parameter string
        self.param_string = ""
        self.param_string = self.param_string + "_dim_"
        for i in range(len(self.latent_dims)):
            self.param_string = self.param_string + str(self.latent_dims[i]) + "_"
        self.param_string = self.param_string[:-1]
        
        # Load and check model_var_idxs
        self.model_var_idxs = catch_list(rom_dict, "model_var_idxs", [[-1]], len_highest=self.num_models)
        for model_idx in range(self.num_models):
            assert self.model_var_idxs[model_idx][0] != -1, "model_var_idxs input incorrectly, probably too few lists"
            # this won't work if we have more than one model
            self.var_idxs = np.array(self.model_var_idxs[0], dtype=np.int32)
            self.num_vars = len(self.var_idxs)
        assert len(self.model_var_idxs) == self.num_models, "Must specify model_var_idxs for every model"

        # get ML library, if requested
        # TODO: only get mllib if method requires it
        self.ml_library = catch_input(rom_dict, "ml_library", "none")
        if self.ml_library != "none":
            self.mllib = get_ml_library(self.ml_library, self)

        # Check model base directory
        if "model_dir" in rom_dict:
            self.model_dir = str(rom_dict["model_dir"])
        else:
            self.model_dir = solver.working_dir
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

        if self.adaptive_rom:
            self.init_singval = np.array([])
            self.init_singval_states = np.array([])
            self.fom_sol_init_window = None
            self.is_intrusive = True

            # Load adaptive ROM inputs
            self.init_window_size = catch_input(rom_dict, "init_window_size", "none")
            self.adeim_update = catch_input(rom_dict, "adeim_update", "AFDEIM")
            self.initbasis_snap_skip = catch_input(rom_dict, "initbasis_snap_skip", "none")
            self.use_fom = catch_input(rom_dict, "use_fom", 0)
            self.adapt_freq = catch_input(rom_dict, "adapt_freq", 1)
            self.update_rank = catch_input(rom_dict, "update_rank", 1)
            self.sampling_update_freq = catch_input(rom_dict, "sampling_update_freq", 1)
            self.num_residual_comp = catch_input(rom_dict, "num_residual_comp", sol_domain.gas_model.num_eqs * sol_domain.mesh.num_cells)

            if self.init_window_size == "none":
                self.initbasis_snap_iter_end = catch_input(rom_dict, "initbasis_snap_iter_end", solver.num_steps) 
            else:
                self.initbasis_snap_iter_end = self.init_window_size
            
            self.initbasis_snap_iter_start = catch_input(rom_dict, "initbasis_snap_iter_start", 0)
        
            if self.initbasis_snap_skip == "none":
                self.initbasis_snap_iter_skip = catch_input(rom_dict, "initbasis_snap_iter_skip", 1)
            else:
                self.initbasis_snap_iter_skip = self.initbasis_snap_skip

            self.initbasis_cent_type = catch_input(rom_dict, "initbasis_cent_type", "mean")
            self.initbasis_norm_type = catch_input(rom_dict, "initbasis_norm_type", "minmax")
        
            # string containing parameters # AADEIM, init window size, window size, update rank, update freq, POD, useFOM, how many residual components
            self.param_string = ""
            self.model_files = [None] * self.num_models
        
            # check if basis and deim files are provided 
            if "model_files" in rom_dict:
                # Load and check model input locations    
                model_files = rom_dict["model_files"]
                assert len(model_files) == self.num_models, "Must provide model_files for each model"
                for model_idx in range(self.num_models):
                    in_file = os.path.join(self.model_dir, model_files[model_idx])
                    assert os.path.isfile(in_file), "Could not find model file at " + in_file
                    self.model_files[model_idx] = in_file
            
                # Load standardization profiles, if they are required
                if rom_dict["rom_method"] == "mplsvt":
                    self.cent_prim_in = catch_list(rom_dict, "cent_prim_in", [""])
                    self.norm_sub_prim_in = catch_list(rom_dict, "norm_sub_prim", [""])
                    self.norm_fac_prim_in = catch_list(rom_dict, "orm_fac_prim", [""])
                    self.cent_cons_in = catch_list(rom_dict, "cent_cons", [""])
                    self.norm_sub_cons_in = catch_list(rom_dict, "norm_sub_cons", [""])
                    self.norm_fac_cons_in = catch_list(rom_dict, "norm_fac_cons", [""])
                else:
                    self.norm_sub_cons_in = catch_list(rom_dict, "norm_sub_cons", [""])
                    self.norm_fac_cons_in = catch_list(rom_dict, "norm_fac_cons", [""])
                    self.cent_cons_in = catch_list(rom_dict, "cent_cons", [""])
            
                if self.hyper_reduc:
                    self.load_hyper_reduc(sol_domain) 
                
            else:
                #if self.rom_method[-7:] == "tfkeras":
                #    raise Exception('Automated computation of basis and DEIM sampling points not yet supported for nonlinear ROMs.')
            
                # compute basis and scaling profiles
                spatial_modes, cent_file, norm_sub_file, norm_fac_file, \
                    self.init_singval, self.init_singval_states, self.fom_sol_init_window, \
                    cent_file_cons, norm_sub_file_cons, norm_fac_file_cons \
                    = gen_rom_basis(self.model_dir, solver.dt, int(self.initbasis_snap_iter_start), \
                    int(self.initbasis_snap_iter_end), int(self.initbasis_snap_iter_skip), self.initbasis_cent_type, \
                    self.initbasis_norm_type, self.model_var_idxs, self.latent_dims, rom_dict["rom_method"])
            
                self.cent_cons_in = cent_file
                self.norm_sub_cons_in = norm_sub_file
                self.norm_fac_cons_in = norm_fac_file
            
                # MPLSVT needs both primitive and conservative variable scaling
                if rom_dict["rom_method"] == "mplsvt":
                    self.cent_prim_in = cent_file
                    self.norm_sub_prim_in = norm_sub_file
                    self.norm_fac_prim_in = norm_fac_file
                    self.cent_cons_in = cent_file_cons
                    self.norm_sub_cons_in = norm_sub_file_cons
                    self.norm_fac_cons_in = norm_fac_file_cons
                else:
                    self.cent_cons_in = cent_file
                    self.norm_sub_cons_in = norm_sub_file
                    self.norm_fac_cons_in = norm_fac_file
            
                for model_idx in range(self.num_models):
                    self.model_files[model_idx] = spatial_modes[model_idx]
                
                # compute hyperreduction sampling points
                if self.hyper_reduc:
                    sampling_id = gen_deim_sampling(self.model_var_idxs, spatial_modes[0], self.latent_dims[0])
                    self.load_hyper_reduc(sol_domain, samp_idx = sampling_id, hyperred_basis = spatial_modes, hyperred_dims = self.latent_dims) 
        
            # Set up hyper-reduction
            self.param_string = self.param_string + "_AADEIM_"
            assert self.hyper_reduc, "Hyper reduction is needed for adaptive basis"
            assert solver.time_scheme == "bdf", "Adaptive basis requires implicit time-stepping"
            assert solver.param_dict['time_order'] == 1, "Adaptive basis rhs evaluation needs backward Euler discretization"
            assert np.abs(np.asarray(self.hyper_reduc_dims) - np.asarray(self.latent_dims)).max() == 0, "ROM and hyperreduction basis dimensions must be the same"
                
            # check that the ROM and hyperreduction bases are the same
            rom_deim_basis_same = 1
            for idx in range(self.num_models):
                if isinstance(self.model_files[idx], np.ndarray):
                    rom_basis = self.model_files[idx]
                else:
                    rom_basis = np.load(self.model_files[idx])
                rom_basis = rom_basis[:,:,:self.latent_dims[idx]]
                    
                if isinstance(self.hyper_reduc_files[idx], np.ndarray):
                    deim_basis = self.hyper_reduc_files[idx]
                else:
                    deim_basis = np.load(self.hyper_reduc_files[idx])
                deim_basis = deim_basis[:,:,:self.hyper_reduc_dims[idx]]
                    
                if not np.allclose(rom_basis, deim_basis):
                    rom_deim_basis_same = 0
                    break
                
            assert rom_deim_basis_same == 1, "ROM and DEIM basis have to be the same."                    
            self.adaptive_rom_window_size = catch_input(rom_dict, "adaptive_rom_window_size", max(self.hyper_reduc_dims)+1)
                    
            self.adaptive_rom_init_time = int(catch_input(rom_dict, "adaptive_rom_init_time", self.initbasis_snap_iter_end)) 
            assert self.adaptive_rom_init_time >= self.adaptive_rom_window_size, "initial window size has to be at least adaptive window size."

            self.adaptive_rom_fom_file = catch_input(rom_dict, "adaptive_rom_fom_file", "unsteady_field_results/sol_cons_FOM_dt_" + str(solver.dt) + ".npy")
            self.adaptive_rom_fom_primfile = catch_input(rom_dict, "adaptive_rom_fom_primfile", "unsteady_field_results/sol_prim_FOM_dt_" + str(solver.dt) + ".npy")
                        
            self.basis_adapted = 0

            assert self.adaptive_rom_init_time < solver.num_steps, "Initial time for adaptive ROM has to be less than the maximum number of time steps!"
                
            self.param_string = self.param_string + "iw_" + str(self.adaptive_rom_init_time)
            self.param_string = self.param_string + "_ws_" + str(self.adaptive_rom_window_size)
            self.param_string = self.param_string + "_uf_" + str(self.sampling_update_freq)
            self.param_string = self.param_string + "_res_" + str(self.num_residual_comp)
            self.param_string = self.param_string + "_usefom_" + str(self.use_fom)
            self.param_string = self.param_string + "_" + self.adeim_update
                
            if solver.out_interval > 1:
                self.param_string = self.param_string + "_skip_" + str(solver.out_interval)
                
            if self.adapt_freq > 1:
                self.param_string = self.param_string + "_af_" + str(self.adapt_freq)
                
            if self.update_rank > 1:
                self.param_string = self.param_string + "_ur_" + str(self.update_rank)

        # Initialize RomMethod, RomVariableMapping, and RomTimeStepper
        # TODO: certain parts in projection_method and space_mapping are coded in a brute force manner to allow them work with the adaptive rom.
        # Adaptive ROM and hyper-reduction should be moved from rom_domain to projection_method.
        self.rom_method = get_rom_method(rom_dict["rom_method"], sol_domain, solver, self)
        self.var_mapping = get_variable_mapping(rom_dict["var_mapping"], sol_domain, self)
        self.time_stepper = get_time_stepper(rom_dict["time_stepper"], sol_domain, solver, self)

        # Initialize RomModel (and associated ROM space mappings)
        self.model_list = [None] * self.num_models
        for model_idx in range(self.num_models):
            self.model_list[model_idx] = RomModel(model_idx, sol_domain, self)
        if self.adaptive_rom:
            assert len(self.model_list) == 1, "AADEIM only works for vector ROM for now."
                
        # Additional initializations specific to RomTimeStepper and RomMethod
        self.time_stepper.init_state(sol_domain, self)
        self.rom_method.init_method(sol_domain, self)

    def advance_time(self, sol_domain, solver):
        """Advance low-dimensional state and full solution forward one physical time iteration.

        For non-intrusive ROMs without a time integrator, simply advances the solution one step.

        For intrusive and non-intrusive ROMs with a time integrator, begins numerical time integration
        and steps through sub-iterations.

        Args:
            sol_domain: SolutionDomain with which this RomDomain is associated.
            solver: SystemSolver containing global simulation parameters.
        """

        # check if basis was adapted
        if self.adaptive_rom:
            if self.basis_adapted == 1:
                # update code and FOM approx with respect to new basis    
                for model in self.model_list:
                    model.code = np.dot(model.space_mapping.trial_basis.T, np.dot(self.rom_method.prev_basis, model.code))
                    model.code_hist[0] = model.code.copy()
                    model.code_hist[1] = model.code.copy()
                    model.space_mapping.decode_sol(model.code)
                
                sol_domain.sol_int.update_state(from_prim=(sol_domain.time_integrator.dual_time))
                sol_domain.sol_int.sol_hist_cons[0] = sol_domain.sol_int.sol_cons.copy()
                sol_domain.sol_int.sol_hist_prim[0] = sol_domain.sol_int.sol_prim.copy()
            
                sol_domain.sol_int.sol_hist_cons[1] = sol_domain.sol_int.sol_cons.copy()
                sol_domain.sol_int.sol_hist_prim[1] = sol_domain.sol_int.sol_prim.copy()

                self.basis_adapted = 0  

        print("Iteration " + str(solver.iter))
        self.time_stepper.advance_iter(sol_domain, solver, self)

        # if adaptive, make adjustments to the stored ROM solution if time iteration is at most initial window size
        # TO DO: This is kept from Wayne's code. It seems the ROM prediction is substituted with direct projection of the FOM solution in the initial window.
        # Double check to make sure this is the right approach.
        if self.adaptive_rom and solver.time_iter <= self.adaptive_rom_init_time :  
            self.rom_code_adaptive_initwindow(solver, sol_domain)

        sol_domain.sol_int.update_sol_hist()
        self.update_code_hist()

        # Depending on the model reduction method, basis is updated in the class corresponding to that method (Gaerkin, LSPG, MPLSVT).
        if self.adaptive_rom:
            self.aadeim(sol_domain, solver)

    def rom_code_adaptive_initwindow(self, solver, sol_domain):
        
        # extract centered and normalized FOM
        fom_sol = self.fom_sol_init_window[:, solver.time_iter]
        
        for model in self.model_list:
            # project FOM solution onto the trial space
            rom_sol = np.dot(model.space_mapping.trial_basis.T, fom_sol)
            
            # update model attributes
            model.code = rom_sol
            model.code_hist[0] = model.code.copy()
            model.space_mapping.decode_sol(model.code)
            
        sol_int = sol_domain.sol_int
            
        # update sol_int attributes
        sol_int.update_state(from_prim=(sol_domain.time_integrator.dual_time))
        sol_int.sol_hist_cons[0] = sol_int.sol_cons.copy()
        sol_int.sol_hist_prim[0] = sol_int.sol_prim.copy()

    def aadeim(self, sol_domain, solver):
        """Update adaptive basis here."""

        # initialize window here
        if solver.time_iter == 1:
                    
            # this is only for testing the code by evaluating the rhs function with the fom solution
            for model_idx, model in enumerate(self.model_list):
                if self.use_fom == 1:
                    model.adapt.load_fom(self, model)
                    
        # this for loop is currently decorative only since the adaptive ROM is not compatible with more than one models in its current shape.
        for model_idx, model in enumerate(self.model_list):
            deim_idx_flat = self.rom_method.direct_samp_idxs_flat
            trial_basis = model.space_mapping.trial_basis
            decoded_rom = model.space_mapping.decode_sol(model.code)
            deim_dim = self.rom_method.hyper_reduc_dim
                    
            # update residual sampling points
            self.rom_method.adapt.update_res_sampling_window(self, solver, sol_domain, trial_basis, deim_idx_flat, decoded_rom, model, self.use_fom)
                    
            # call adeim to update the POD basis
            if self.rom_method.adapt.fcn_window.shape[1] >= self.adaptive_rom_window_size and solver.time_iter > self.adaptive_rom_init_time and solver.time_iter % self.adapt_freq == 0:
                        
                #if self.adeim_update != "POD": #self.adaptiveROMADEIMadapt == "ADEIM" or self.adaptiveROMADEIMadapt == "AODEIM":
                updated_basis, updated_interp_pts = self.rom_method.adapt.adeim(self, trial_basis, deim_idx_flat, deim_dim, sol_domain.mesh.num_cells, solver, model.code)
                #else:    
                    # this is only for testing, so it can be removed from the final version
                    #updated_basis, updated_interp_pts = self.rom_method.adapt.pod_basis(deim_dim, sol_domain.mesh.num_cells, trial_basis, solver)

                # update deim interpolation points
                # update rom_domain and sol_domain attributes. call method below to update rest
                self.direct_samp_idxs = updated_interp_pts
                sol_domain.direct_samp_idxs = updated_interp_pts
                self.rom_method.flatten_deim_idxs(self, sol_domain)

                # update basis. make sure to update the deim basis too
                self.rom_method.update_basis(updated_basis, self)
                
        if self.rom_method.adapt.fcn_window.shape[1] >= self.adaptive_rom_window_size and solver.time_iter > self.adaptive_rom_init_time and solver.time_iter % self.adapt_freq == 0:
            self.compute_cellidx_hyper_reduc(sol_domain)
            self.basis_adapted = 1

    def update_code_hist(self):
        """Update low-dimensional state history after physical time step."""

        for model in self.model_list:
            model.code_hist[1:] = model.code_hist[:-1]
            model.code_hist[0] = model.code.copy()


    # TO DO: This function and the next one should be moved to projection_method. Hyper-reduction is disabled in the main branch!
    def load_hyper_reduc(self, sol_domain, samp_idx = [], hyperred_basis = [], hyperred_dims = []):
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

        if not isinstance(samp_idx, np.ndarray):
            # load and check sample points
            samp_file = catch_input(self.rom_dict, "samp_file", "")
            assert samp_file != "", "Must supply samp_file if performing hyper-reduction"
            samp_file = os.path.join(self.model_dir, samp_file)
            assert os.path.isfile(samp_file), "Could not find samp_file at " + samp_file
    
            # Indices of directly sampled cells, within sol_prim/cons
            # NOTE: assumed that sample indices are zero-indexed
            sol_domain.direct_samp_idxs = np.load(samp_file).flatten()
        else:
            sol_domain.direct_samp_idxs = samp_idx.flatten()
            
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
        
        # Paths to hyper-reduction files (unpacked later)
        self.hyper_reduc_files = [None] * self.num_models
        if hyperred_basis == []:
            hyper_reduc_files = self.rom_dict["hyper_reduc_files"]
            assert len(hyper_reduc_files) == self.num_models, "Must provide hyper_reduc_files for each model"
            for model_idx in range(self.num_models):
                in_file = os.path.join(self.model_dir, hyper_reduc_files[model_idx])
                assert os.path.isfile(in_file), "Could not find hyper-reduction file at " + in_file
                self.hyper_reduc_files[model_idx] = in_file
        else:
            for model_idx in range(self.num_models):
                self.hyper_reduc_files[model_idx] = hyperred_basis[model_idx]

        # Load hyper reduction dimensions and check validity
        if hyperred_dims != []:
            self.hyper_reduc_dims = hyperred_dims
        else:
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

        # Copy indices for ease of use
        self.num_samp_cells = sol_domain.num_samp_cells
        self.direct_samp_idxs = sol_domain.direct_samp_idxs

        self.compute_cellidx_hyper_reduc(sol_domain)

    def compute_cellidx_hyper_reduc(self, sol_domain):  
        
        # moved part of load_hyper_reduc here so that this function can be called if DEIM interpolation points are adapted
        
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

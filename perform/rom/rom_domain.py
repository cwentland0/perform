import os
from time import sleep

import numpy as np

from perform.constants import REAL_TYPE
from perform.input_funcs import read_input_file, catch_list, catch_input
from perform.solution.solution_phys import SolutionPhys
from perform.time_integrator import get_time_integrator
from perform.rom import get_rom_model

# TODO: when moving to multi-domain, it may be useful to just
# 	hold a sol_domain inside RomDomain for the associated full-dim solution
# 	Still a pain to move around since it's associated with the RomDomain
# 	and not the romModel, but whatever
# TODO: need to eliminate normSubProf, just roll it into centProf (or reverse)


class RomDomain:
	"""
	Container class for ROM parameters and romModels
	"""

	def __init__(self, sol_domain, solver):

		rom_dict = read_input_file(solver.rom_inputs)
		self.rom_dict = rom_dict

		# load model parameters
		self.rom_method = str(rom_dict["rom_method"])
		self.num_models = int(rom_dict["num_models"])
		self.latent_dims = \
			catch_list(rom_dict, "latent_dims", [0], len_highest=self.num_models)
		model_var_idxs = \
			catch_list(rom_dict, "model_var_idxs", [[-1]], len_highest=self.num_models)

		# check model parameters
		for i in self.latent_dims:
			assert (i > 0), "latent_dims must contain positive integers"

		if self.num_models == 1:
			assert (len(self.latent_dims) == 1), \
				"Must provide only one value of latent_dims when num_models = 1"
			assert (self.latent_dims[0] > 0), \
				"latent_dims must contain positive integers"
		else:
			if len(self.latent_dims) == self.num_models:
				pass
			elif len(self.latent_dims) == 1:
				print("Only one value provided in latent_dims, applying to all models")
				sleep(1.0)
				self.latent_dims = [self.latent_dims[0]] * self.num_models
			else:
				raise ValueError("Must provide either num_models"
								+ "or 1 entry in latent_dims")

		# load and check model_var_idxs
		for model_idx in range(self.num_models):
			assert (model_var_idxs[model_idx][0] != -1), \
				"model_var_idxs input incorrectly, probably too few lists"
		assert (len(model_var_idxs) == self.num_models), \
			"Must specify model_var_idxs for every model"
		model_var_sum = 0
		for model_idx in range(self.num_models):
			model_var_sum += len(model_var_idxs[model_idx])
			for model_var_idx in model_var_idxs[model_idx]:
				assert (model_var_idx >= 0), \
					"model_var_idxs must be non-negative integers"
				assert (model_var_idx < sol_domain.gas_model.num_eqs), \
					"model_var_idxs must less than the number of governing equations"
		assert (model_var_sum == sol_domain.gas_model.num_eqs), \
			("Must specify as many model_var_idxs entries as governing equations ("
			+ str(model_var_sum) + " != " + str(sol_domain.gas_model.num_eqs) + ")")
		model_var_idxs_one_list = sum(model_var_idxs, [])
		assert (len(model_var_idxs_one_list) == len(set(model_var_idxs_one_list))), \
			"All entries in model_var_idxs must be unique"
		self.model_var_idxs = model_var_idxs

		# load and check model input locations
		self.model_dir = str(rom_dict["model_dir"])
		model_files = rom_dict["model_files"]
		self.model_files = [None] * self.num_models
		assert (len(model_files) == self.num_models), \
			"Must provide model_files for each model"
		for model_idx in range(self.num_models):
			in_file = os.path.join(self.model_dir, model_files[model_idx])
			assert (os.path.isfile(in_file)), \
				"Could not find model file at " + in_file
			self.model_files[model_idx] = in_file

		# load standardization profiles, if they are required
		self.cent_ic = catch_input(rom_dict, "cent_ic", False)
		self.norm_sub_cons_in = catch_list(rom_dict, "norm_sub_cons_in", [""])
		self.norm_fac_cons_in = catch_list(rom_dict, "norm_fac_cons_in", [""])
		self.cent_cons_in = catch_list(rom_dict, "cent_cons_in", [""])
		self.norm_sub_prim_in = catch_list(rom_dict, "norm_sub_prim_in", [""])
		self.norm_fac_prim_in = catch_list(rom_dict, "norm_fac_prim_in", [""])
		self.cent_prim_in = catch_list(rom_dict, "cent_prim_in", [""])

		# load low-dimensional initial condition state, if desired
		self.load_init_code(rom_dict)

		self.set_model_flags()

		self.adaptive_rom = catch_input(rom_dict, "adaptive_rom", False)

		# set up hyper-reduction, if necessary
		self.hyper_reduc = catch_input(rom_dict, "hyper_reduc", False)
		if self.is_intrusive and self.hyper_reduc:
			self.load_hyper_reduc(sol_domain)

		# get time integrator, if necessary
		# TODO: time_scheme should be specific to the RomDomain, not the solver
		if self.has_time_integrator:
			self.time_integrator = \
				get_time_integrator(solver.time_scheme, solver.param_dict)
		else:
			self.time_integrator = None 	# TODO: this might be pointless

		# initialize models for domain
		self.model_list = [None] * self.num_models
		for model_idx in range(self.num_models):

			self.model_list[model_idx] = \
				get_rom_model(model_idx, self, sol_domain)
			model = self.model_list[model_idx]

			# initialize state
			if self.init_rom_from_file[model_idx]:
				model.init_from_code(self.code_init[model_idx], sol_domain)
			else:
				model.init_from_sol(sol_domain)

			# initialize code history
			# TODO: this is necessary for non-time-integrated methods, e.g. TCN
			model.code_hist = \
				[model.code.copy()] * (self.time_integrator.time_order + 1)

		sol_domain.sol_int.update_state(from_cons=self.target_cons)

		# overwrite history with initialized solution
		sol_domain.sol_int.sol_hist_cons = \
			[sol_domain.sol_int.sol_cons.copy()] * (self.time_integrator.time_order + 1)
		sol_domain.sol_int.sol_hist_prim = \
			[sol_domain.sol_int.sol_prim.copy()] * (self.time_integrator.time_order + 1)

	def set_model_flags(self):
		"""
		Set universal ROM method flags that dictate various execution behaviors
		"""

		# TODO: move this to __init__.py

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
		assert (self.target_cons != self.target_prim), \
			"Model must target either the primitive or conservative variables"

	def load_init_code(self, rom_dict):
		"""
		Load low-dimensional state from disk if provided,
		overwriting full-dimensional state

		If not provided, ROM model will attempt to compute
		initial low-dimensional state from initial full-dimensional state
		"""

		self.low_dim_init_files = catch_list(rom_dict, "low_dim_init_files", [""])

		len_init_files = len(self.low_dim_init_files)
		self.init_rom_from_file = [False] * self.num_models
		self.code_init = [None] * self.num_models

		# no init files given
		if (len_init_files == 1) and (self.low_dim_init_files[0] == ""):
			pass

		# TODO: this format leads to some messy storage,
		# 	figure out a better way. Maybe .npz files?
		else:
			assert (len_init_files == self.num_models), \
				("If initializing any ROM model from a file,"
				+ "must provide list entries for every model. "
				+ "If you don't wish to initialize from file for a model,"
				+ " input an empty string \"\" in the list entry.")
			for model_idx in range(self.num_models):
				init_file = self.low_dim_init_files[model_idx]
				if (init_file != ""):
					assert (os.path.isfile(init_file)), \
						"Could not find ROM initialization file at " + init_file
					self.code_init[model_idx] = np.load(init_file)
					self.init_rom_from_file[model_idx] = True

	def load_hyper_reduc(self, sol_domain):
		"""
		Loads direct sampling indices and
		determines cell indices for calculating fluxes and gradients
		"""

		raise ValueError("Hyper-reduction temporarily out of commission for development")

		# TODO: add some explanations for what each index array accomplishes

		# load and check sample points
		samp_file = catch_input(self.rom_dict, "samp_file", "")
		assert (samp_file != ""), \
			"Must supply samp_file if performing hyper-reduction"
		samp_file = os.path.join(self.model_dir, samp_file)
		assert (os.path.isfile(samp_file)), \
			("Could not find samp_file at " + samp_file)

		# NOTE: assumed that sample indices are zero-indexed
		sol_domain.direct_samp_idxs = np.load(samp_file).flatten()
		sol_domain.direct_samp_idxs = \
			(np.sort(sol_domain.direct_samp_idxs)).astype(np.int32)
		sol_domain.num_samp_cells = len(sol_domain.direct_samp_idxs)
		assert (sol_domain.num_samp_cells
				<= sol_domain.mesh.num_cells), \
			"Cannot supply more sampling points than cells in domain."
		assert (np.amin(sol_domain.direct_samp_idxs)
				>= 0), \
			"Sampling indices must be non-negative integers"
		assert (np.amax(sol_domain.direct_samp_idxs)
				< sol_domain.mesh.num_cells), \
			"Sampling indices must be less than the number of cells in the domain"
		assert (len(np.unique(sol_domain.direct_samp_idxs))
				== sol_domain.num_samp_cells), \
			"Sampling indices must be unique"

		# TODO: should probably shunt these over to a function
		# 	for when indices get updated in adaptive method

		# compute indices for inviscid flux calculations
		# NOTE: have to account for fact that boundary cells are prepended/appended
		sol_domain.flux_samp_left_idxs = \
			np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
		sol_domain.flux_samp_left_idxs[0::2] = sol_domain.direct_samp_idxs
		sol_domain.flux_samp_left_idxs[1::2] = sol_domain.direct_samp_idxs + 1

		sol_domain.flux_samp_right_idxs = \
			np.zeros(2 * sol_domain.num_samp_cells, dtype=np.int32)
		sol_domain.flux_samp_right_idxs[0::2] = sol_domain.direct_samp_idxs + 1
		sol_domain.flux_samp_right_idxs[1::2] = sol_domain.direct_samp_idxs + 2

		# eliminate repeated indices
		sol_domain.flux_samp_left_idxs = np.unique(sol_domain.flux_samp_left_idxs)
		sol_domain.flux_samp_right_idxs = np.unique(sol_domain.flux_samp_right_idxs)
		sol_domain.num_flux_faces = len(sol_domain.flux_samp_left_idxs)

		# Roe average
		if sol_domain.invisc_flux_name == "roe":
			ones_prof = \
				np.ones((sol_domain.gas_model.num_eqs, sol_domain.num_flux_faces),
						dtype=const.REAL_TYPE)
			sol_domain.sol_ave = solutionPhys(sol_domain.gas_model,
												sol_domain.num_flux_faces,
												solPrimIn=ones_prof)

		# to slice flux when calculating RHS
		sol_domain.flux_rhs_idxs = np.zeros(sol_domain.num_samp_cells, np.int32)
		for i in range(1, sol_domain.num_samp_cells):
			if (sol_domain.direct_samp_idxs[i]
					== (sol_domain.direct_samp_idxs[i - 1] + 1)):
				sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 1
			else:
				sol_domain.flux_rhs_idxs[i] = sol_domain.flux_rhs_idxs[i - 1] + 2

		# compute indices for gradient calculations
		# NOTE: also need to account for prepended/appended boundary cells
		# TODO: generalize for higher-order schemes
		if sol_domain.space_order > 1:
			if sol_domain.space_order == 2:
				sol_domain.grad_idxs = \
					np.concatenate((sol_domain.direct_samp_idxs + 1,
									sol_domain.direct_samp_idxs,
									sol_domain.direct_samp_idxs + 2))
				sol_domain.grad_idxs = np.unique(sol_domain.grad_idxs)
				# exclude left neighbor of inlet, right neighbor of outlet
				if sol_domain.grad_idxs[0] == 0:
					sol_domain.grad_idxs = sol_domain.grad_idxs[1:]
				if sol_domain.grad_idxs[-1] == (sol_domain.mesh.num_cells + 1):
					sol_domain.grad_idxs = sol_domain.grad_idxs[:-1]
				sol_domain.num_grad_cells = len(sol_domain.grad_idxs)

				# neighbors of gradient cells
				sol_domain.grad_neigh_idxs = \
					np.concatenate((sol_domain.grad_idxs - 1, sol_domain.grad_idxs + 1))
				sol_domain.grad_neigh_idxs = np.unique(sol_domain.grad_neigh_idxs)
				# exclude left neighbor of inlet, right neighbor of outlet
				if sol_domain.grad_neigh_idxs[0] == -1:
					sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[1:]
				if (sol_domain.grad_neigh_idxs[-1] == (sol_domain.mesh.num_cells + 2)):
					sol_domain.grad_neigh_idxs = sol_domain.grad_neigh_idxs[:-1]

				# indices of grad_idxs in grad_neigh_idxs
				_, _, sol_domain.grad_neighextract = \
					np.intersect1d(sol_domain.grad_idxs,
									sol_domain.grad_neigh_idxs,
									return_indices=True)

				# indices of grad_idxs in flux_samp_left_idxs and flux_samp_right_idxs
				# 	and vice versa
				_, sol_domain.grad_left_extract, sol_domain.flux_left_extract = \
					np.intersect1d(sol_domain.grad_idxs,
									sol_domain.flux_samp_left_idxs,
									return_indices=True)
				_, sol_domain.grad_right_extract, sol_domain.flux_right_extract = \
					np.intersect1d(sol_domain.grad_idxs,
									sol_domain.flux_samp_right_idxs,
									return_indices=True)

			else:
				raise ValueError("Sampling for higher-order schemes not implemented yet")

		# copy indices for ease of use
		self.num_samp_cells = sol_domain.num_samp_cells
		self.direct_samp_idxs = sol_domain.direct_samp_idxs

		# paths to hyper-reduction files (unpacked later)
		hyper_reduc_files = self.rom_dict["hyper_reduc_files"]
		self.hyper_reduc_files = [None] * self.num_models
		assert (len(hyper_reduc_files) == self.num_models), \
			"Must provide hyper_reduc_files for each model"
		for model_idx in range(self.num_models):
			in_file = os.path.join(self.model_dir, hyper_reduc_files[model_idx])
			assert (os.path.isfile(in_file)), \
				"Could not find hyper-reduction file at " + in_file
			self.hyper_reduc_files[model_idx] = in_file

		# load hyper reduction dimensions and check validity
		self.hyper_reduc_dims = \
			catch_list(self.rom_dict, "hyper_reduc_dims",
						[0], len_highest=self.num_models)
		for i in self.hyper_reduc_dims:
			assert (i > 0), "hyper_reduc_dims must contain positive integers"
		if self.num_models == 1:
			assert (len(self.hyper_reduc_dims) == 1), \
				"Must provide only one value of hyper_reduc_dims when num_models = 1"
			assert (self.hyper_reduc_dims[0] > 0), \
				"hyper_reduc_dims must contain positive integers"
		else:
			if (len(self.hyper_reduc_dims) == self.num_models):
				pass
			elif (len(self.hyper_reduc_dims) == 1):
				print("Only one value provided in hyper_reduc_dims,"
						+ " applying to all models")
				sleep(1.0)
				self.hyper_reduc_dims = [self.hyper_reduc_dims[0]] * self.num_models
			else:
				raise ValueError("Must provide either num_models"
								+ " or 1 entry in hyper_reduc_dims")

	def advance_iter(self, sol_domain, solver):
		"""
		Advance low-dimensional state forward one time iteration
		"""

		print("Iteration " + str(solver.iter))

		# update model which does NOT require numerical time integration
		if not self.has_time_integrator:
			raise ValueError("Iteration advance for models without"
							+ " numerical time integration not yet implemented")

		# if method requires numerical time integration
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
		"""
		Advance physical solution forward one subiteration of time integrator
		"""

		sol_int = sol_domain.sol_int
		res, res_jacob = None, None

		if self.is_intrusive:
			sol_domain.calc_rhs(solver)

		if self.time_integrator.time_type == "implicit":

			# compute residual and residual Jacobian
			if self.is_intrusive:
				res = self.time_integrator.calc_residual(sol_int.sol_hist_cons,
														sol_int.rhs,
														solver)
				res_jacob = sol_domain.calc_res_jacob(solver)

			# compute change in low-dimensional state
			for model_idx, model in enumerate(self.model_list):
				d_code, code_lhs, code_rhs = model.calc_d_code(res_jacob, res, sol_domain)
				model.code += d_code
				model.code_hist[0] = model.code.copy()
				model.update_sol(sol_domain)

				# compute ROM residual for convergence measurement
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
		"""
		Update low-dimensional state history after physical time step
		"""

		for model in self.model_list:

			model.code_hist[1:] = model.code_hist[:-1]
			model.code_hist[0] = model.code.copy()

	def calc_code_res_norms(self, sol_domain, solver, subiter):
		"""
		Calculate and print linear solve residual norms
		Note that output is ORDER OF MAGNITUDE of residual norm
		(i.e. 1e-X, where X is the order of magnitude)
		"""

		# compute residual norm for each model
		norm_l2_sum = 0.0
		norm_l1_sum = 0.0
		for model in self.model_list:
			norm_l2, norm_l1 = model.calc_code_norms()
			norm_l2_sum += norm_l2
			norm_l1_sum += norm_l1

		# average over all models
		norm_l2 = norm_l2_sum / self.num_models
		norm_l1 = norm_l1_sum / self.num_models

		# norm is sometimes zero, just default to -16 I guess
		if norm_l2 == 0.0:
			norm_out_l2 = -16.0
		else:
			norm_out_l2 = np.log10(norm_l2)

		if norm_l1 == 0.0:
			norm_out_l1 = -16.0
		else:
			norm_out_l1 = np.log10(norm_l1)

		out_string = ((str(subiter + 1)
						+ ":\tL2: %18.14f, \tL1: %18.14f")
						% (norm_out_l2, norm_out_l1))
		print(out_string)

		sol_domain.sol_int.res_norm_l2 = norm_l2
		sol_domain.sol_int.resNormL1 = norm_l1
		sol_domain.sol_int.res_norm_history[solver.iter - 1, :] = [norm_l2, norm_l1]

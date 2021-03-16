import os

import numpy as np

from perform.constants import REAL_TYPE


class RomModel:
	"""
	Base class for any ROM model

	Assumed that every model may be equipped with
	some sort of data standardization requirement

	Also assumed that every model has some means of
	computing the full-dimensional state from the low
	dimensional state, i.e. a "decoder"
	"""

	def __init__(self, model_idx, rom_domain, solver, sol_domain):

		self.model_idx = model_idx
		self.latent_dim = rom_domain.latent_dims[self.model_idx]
		self.var_idxs = np.array(rom_domain.model_var_idxs[self.model_idx],
									dtype=np.int32)
		self.num_vars = len(self.var_idxs)
		self.num_cells = solver.mesh.num_cells
		self.sol_shape = (self.num_vars, self.num_cells)

		# Just copy some stuff for less clutter
		self.model_dir = rom_domain.model_dir
		self.target_cons = rom_domain.target_cons
		self.hyper_reduc = rom_domain.hyper_reduc

		self.code = np.zeros(self.latent_dim, dtype=REAL_TYPE)
		self.res = np.zeros(self.latent_dim, dtype=REAL_TYPE)

		# Get normalization profiles, if necessary
		self.norm_sub_prof_cons = None
		self.norm_sub_prof_prim = None
		self.norm_fac_prof_cons = None
		self.norm_fac_prof_prim = None
		self.cent_prof_cons = None
		self.cent_prof_prim = None
		if rom_domain.hasConsNorm:
			self.norm_sub_prof_cons = \
				self.load_standardization(
					os.path.join(self.model_dir, rom_domain.norm_sub_cons_in[self.model_idx]),
					default="zeros"
				)
			self.norm_fac_prof_cons = \
				self.load_standardization(
					os.path.join(self.model_dir, rom_domain.norm_fac_cons_in[self.model_idx]),
					default="ones"
				)
		if rom_domain.hasPrimNorm:
			self.norm_sub_prof_prim = \
				self.load_standardization(
					os.path.join(self.model_dir, rom_domain.norm_sub_prim_in[self.model_idx]),
					default="zeros"
				)
			self.norm_fac_prof_prim = \
				self.load_standardization(
					os.path.join(self.model_dir, rom_domain.norm_fac_prim_in[self.model_idx]),
					default="ones"
				)

		# Get centering profiles, if necessary
		# If cent_ic, just use given initial conditions
		if rom_domain.has_cons_cent:
			if rom_domain.cent_ic:
				self.cent_prof_cons = sol_domain.sol_int.sol_cons[self.var_idxs, :].copy()
			else:
				self.cent_prof_cons = \
					self.load_standardization(
						os.path.join(self.model_dir, rom_domain.cent_cons_in[self.model_idx]),
						default="zeros"
					)
		if rom_domain.has_prim_cent:
			if rom_domain.cent_ic:
				self.cent_prof_prim = sol_domain.sol_int.sol_prim[self.var_idxs, :].copy()
			else:
				self.cent_prof_prim = \
					self.load_standardization(
						os.path.join(self.model_dir, rom_domain.cent_prim_in[self.model_idx]),
						default="zeros"
					)

	def load_standardization(self, stand_input, default="zeros"):

		try:
			# TODO: add ability to accept single scalar value for stand_input
			# 		catchList doesn't handle this when loading normSubIn, etc.

			# Load single complete standardization profile from file
			stand_prof = np.load(stand_input)
			assert (stand_prof.shape == self.sol_shape)
			return stand_prof

		except AssertionError:
			print("Standardization profile at " + stand_input
					+ " did not match solution shape")

		if default == "zeros":
			print("WARNING: standardization load failed or not specified,"
					+ " defaulting to zeros")
			stand_prof = np.zeros(self.sol_shape, dtype=REAL_TYPE)
		elif default == "ones":
			print("WARNING: standardization load failed or not specified,"
					+ " defaulting to ones")
			stand_prof = np.zeros(self.sol_shape, dtype=REAL_TYPE)

		return stand_prof

	def standardize_data(self, arr,
						normalize=True, norm_fac_prof=None, norm_sub_prof=None,
						center=True, cent_prof=None,
						inverse=False):
		"""
		(de)centering and (de)normalization
		"""

		if normalize:
			assert (norm_fac_prof is not None), \
				"Must provide normalization division factor to normalize"
			assert (norm_sub_prof is not None), \
				"Must provide normalization subtractive factor to normalize"
		if center:
			assert (cent_prof is not None), \
				"Must provide centering profile to center"

		if inverse:
			if normalize:
				arr = self.normalize(arr, norm_fac_prof,
									norm_sub_prof, denormalize=True)
			if center:
				arr = self.center(arr, cent_prof, decenter=True)
		else:
			if center:
				arr = self.center(arr, cent_prof, decenter=False)
			if normalize:
				arr = self.normalize(arr, norm_fac_prof,
									norm_sub_prof, denormalize=False)

		return arr

	def center(self, arr, cent_prof, decenter=False):
		"""
		(de)center input vector according to loaded centering profile
		"""

		if decenter:
			arr += cent_prof
		else:
			arr -= cent_prof
		return arr

	def normalize(self, arr, norm_fac_prof, norm_sub_prof, denormalize=False):
		"""
		(De)normalize input vector according to
		subtractive and divisive normalization profiles
		"""

		if denormalize:
			arr = arr * norm_fac_prof + norm_sub_prof
		else:
			arr = (arr - norm_sub_prof) / norm_fac_prof
		return arr

	def decode_sol(self, code_in):
		"""
		Compute full decoding of solution, including decentering and denormalization
		"""

		sol = self.apply_decoder(code_in)

		if self.target_cons:
			sol = self.standardize_data(sol,
										normalize=True,
										norm_fac_prof=self.norm_fac_prof_cons,
										norm_sub_prof=self.norm_sub_prof_cons,
										center=True, cent_prof=self.cent_prof_cons,
										inverse=True)
		else:
			sol = self.standardize_data(sol,
										normalize=True,
										norm_fac_prof=self.norm_fac_prof_prim,
										norm_sub_prof=self.norm_sub_prof_prim,
										center=True, cent_prof=self.cent_prof_prim,
										inverse=True)

		return sol

	def init_from_code(self, code0, sol_domain):
		"""
		Initialize full-order solution from input low-dimensional state
		"""

		self.code = code0.copy()

		if self.target_cons:
			sol_domain.sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)
		else:
			sol_domain.sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)

	def update_sol(self, sol_domain):
		"""
		Update solution after code has been updated
		"""

		# TODO: could just use this to replace init_from_code?

		if self.target_cons:
			sol_domain.sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)
		else:
			sol_domain.sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)

	def calc_code_norms(self):
		"""
		Compute L1 and L2 norms of low-dimensional state linear solve residuals
		Scaled by number of elements, so "L2 norm" here is really RMS
		"""

		res_abs = np.abs(self.res)

		# L2 norm
		res_norm_l2 = np.sum(np.square(res_abs))
		res_norm_l2 /= self.latent_dim
		res_norm_l2 = np.sqrt(res_norm_l2)

		# L1 norm
		res_norm_l1 = np.sum(res_abs)
		res_norm_l1 /= self.latent_dim

		return res_norm_l2, res_norm_l1

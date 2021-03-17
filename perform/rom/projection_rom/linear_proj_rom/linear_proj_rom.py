import numpy as np

from perform.rom.projection_rom.projection_rom import ProjectionROM


class LinearProjROM(ProjectionROM):
	"""
	Base class for all projection-based ROMs
	which use a linear basis representation
	"""

	def __init__(self, model_idx, rom_domain, sol_domain):

		super().__init__(model_idx, rom_domain, sol_domain)

		# load and check trial basis
		self.trial_basis = np.load(rom_domain.model_files[self.model_idx])
		num_vars_basis_in, num_cells_basis_in, num_modes_basis_in = \
			self.trial_basis.shape

		assert (num_vars_basis_in == self.num_vars), \
			("Basis at " + rom_domain.model_files[self.model_idx]
			+ " represents a different number of variables "
			+ "than specified by modelVarIdxs ("
			+ str(num_vars_basis_in) + " != " + str(self.num_vars) + ")")
		assert (num_cells_basis_in == sol_domain.mesh.num_cells), \
			("Basis at " + rom_domain.model_files[self.model_idx]
			+ " has a different number of cells than the physical domain ("
			+ str(num_cells_basis_in) + " != " + str(sol_domain.mesh.num_cells) + ")")
		assert (num_modes_basis_in >= self.latent_dim), \
			("Basis at " + rom_domain.model_files[self.model_idx]
			+ " must have at least " + str(self.latent_dim) + " modes ("
			+ str(num_modes_basis_in) + " < " + str(self.latent_dim) + ")")

		# flatten first two dimensions for easier matmul
		self.trial_basis = self.trial_basis[:, :, :self.latent_dim]
		self.trial_basis = \
			np.reshape(self.trial_basis, (-1, self.latent_dim), order='C')

		# load and check gappy POD basis
		if rom_domain.hyper_reduc:
			hyper_reduc_basis = np.load(rom_domain.hyper_reduc_files[self.model_idx])
			assert (hyper_reduc_basis.ndim == 3), \
				"Hyper-reduction basis must have three axes"
			assert (hyper_reduc_basis.shape[:2]
					== (sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells)), \
				"Hyper reduction basis must have shape [num_eqs, num_cells, numHRModes]"

			self.hyper_reduc_dim = rom_domain.hyper_reduc_dims[self.model_idx]
			hyper_reduc_basis = hyper_reduc_basis[:, :, :self.hyper_reduc_dim]
			self.hyper_reduc_basis = \
				np.reshape(hyper_reduc_basis, (-1, self.hyper_reduc_dim), order="C")

			# indices for sampling flattened hyper_reduc_basis
			self.direct_hyper_reduc_samp_idxs = \
				np.zeros(rom_domain.num_samp_cells * self.num_vars, dtype=np.int32)
			for var_num in range(self.num_vars):
				idx1 = var_num * rom_domain.num_samp_cells
				idx2 = (var_num + 1) * rom_domain.num_samp_cells
				self.direct_hyper_reduc_samp_idxs[idx1:idx2] = \
					rom_domain.direct_samp_idxs + var_num * sol_domain.mesh.num_cells

	def init_from_sol(self, sol_domain):
		"""
		Initialize full-order solution from projection of
		loaded full-order initial conditions
		"""

		if self.target_cons:
			sol = \
				self.standardize_data(
					sol_domain.sol_int.sol_cons[self.var_idxs, :],
					normalize=True,
					norm_fac_prof=self.norm_fac_prof_cons,
					norm_sub_prof=self.norm_sub_prof_cons,
					center=True, cent_prof=self.cent_prof_cons,
					inverse=False
				)
			self.code = self.project_to_low_dim(self.trial_basis, sol, transpose=True)
			sol_domain.sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)

		else:
			sol = \
				self.standardize_data(
					sol_domain.sol_int.sol_prim[self.var_idxs, :],
					normalize=True,
					norm_fac_prof=self.norm_fac_prof_prim,
					norm_sub_prof=self.norm_sub_prof_prim,
					center=True, cent_prof=self.cent_prof_prim,
					inverse=False
				)
			self.code = self.project_to_low_dim(self.trial_basis, sol, transpose=True)
			sol_domain.sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)

	def apply_decoder(self, code):
		"""
		Compute raw decoding of code, without de-normalizing or de-centering
		"""

		sol = self.trial_basis @ code
		sol = np.reshape(sol, (self.num_vars, -1), order="C")
		return sol

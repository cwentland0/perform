import numpy as np

from perform.rom.rom_model import RomModel
from perform.constants import REAL_TYPE


class ProjectionROM(RomModel):
	"""
	Base class for projection-based reduced-order model

	This model makes no assumption on the form of the decoder,
	but assumes a linear projection onto the low-dimensional space
	"""

	def __init__(self, modelIdx, rom_domain, solver, sol_domain):

		super().__init__(modelIdx, rom_domain, solver, sol_domain)

	def project_to_low_dim(self, projector, full_dim_arr, transpose=False):
		"""
		Project given full-dimensional vector onto low-dimensional
		space via given projector

		Assumed that full_dim_arr is either 1D array or is in
		[numVars, numCells] order

		Assumed that projector is already in
		[numModes, numVars x numCells] order
		"""

		if (full_dim_arr.ndim == 2):
			full_dim_vec = full_dim_arr.flatten(order="C")
		elif (full_dim_arr.ndim == 1):
			full_dim_vec = full_dim_arr.copy()
		else:
			raise ValueError("full_dim_arr must be one- or two-dimensional")

		if transpose:
			code_out = projector.T @ full_dim_vec
		else:
			code_out = projector @ full_dim_vec

		return code_out

	def calc_rhs_low_dim(self, rom_domain, sol_domain):
		"""
		Project RHS onto low-dimensional space for explicit time integrators

		Assumes that RHS term is scaled using an appropriate conservative
		variable normalization profile
		"""

		# scale RHS
		norm_sub_prof = np.zeros(self.norm_fac_prof_cons.shape, dtype=REAL_TYPE)
		rhs_scaled = \
			self.standardize_data(
				sol_domain.sol_int.rhs[self.var_idxs[:, None],
				sol_domain.direct_samp_idxs[None, :]],
				normalize=True,
				norm_fac_prof=self.norm_fac_prof_cons[:, sol_domain.direct_samp_idxs],
				norm_sub_prof=norm_sub_prof[:, sol_domain.direct_samp_idxs],
				center=False, inverse=False
			)

		# calc projection operator and project
		self.calc_projector(sol_domain)
		self.rhs_low_dim = self.project_to_low_dim(self.projector, rhs_scaled,
													transpose=False)

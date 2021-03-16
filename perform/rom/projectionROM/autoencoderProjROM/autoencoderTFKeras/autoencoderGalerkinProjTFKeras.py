import numpy as np
import tensorflow as tf
from scipy.linalg import pinv

from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import AutoencoderTFKeras


class AutoencoderGalerkinProjTFKeras(AutoencoderTFKeras):
	"""
	Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, model_idx, rom_domain, solver, sol_domain):

		if rom_domain.time_integrator.dual_time:
			raise ValueError("Galerkin is intended for conservative"
							+ "variable evolution, please set dual_time = False")

		super().__init__(model_idx, rom_domain, solver, sol_domain)

	def calc_projector(self, sol_domain):
		"""
		Compute rhs projection operator
		Decoder projector is pseudo-inverse of decoder Jacobian
		Encoder projector is just encoder Jacobian
		"""

		jacob = self.calc_model_jacobian(sol_domain)

		if self.encoder_jacob:
			self.projector = jacob

		else:
			self.projector = pinv(jacob)

	def calc_d_code(self, res_jacob, res, sol_domain):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		TODO: this is non-general and janky, only valid for BDF
		"""

		jacob = self.calc_model_jacobian(sol_domain)

		if (self.encoder_jacob):
			jacob_pinv = jacob * self.norm_fac_prof_cons.ravel(order="C")[None, :]

		else:
			scaled_jacob = jacob * self.norm_fac_prof_cons.ravel(order="C")[:, None]
			jacob_pinv = pinv(scaled_jacob)

		# Newton iteration linear solve
		lhs = (
			jacob_pinv @ (res_jacob.toarray()
			/ self.norm_fac_prof_cons.ravel(order="C")[:, None])
			@ scaled_jacob
		)
		rhs = (
			jacob_pinv @ (res
			/ self.norm_fac_prof_cons).ravel(order="C")
		)

		d_code = np.linalg.solve(lhs, rhs)

		return d_code, lhs, rhs

import numpy as np
import tensorflow as tf

from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import AutoencoderTFKeras


class AutoencoderLSPGProjTFKeras(AutoencoderTFKeras):
	"""
	Class for computing non-linear least-squares Petrov-Galerkin ROMs
	via a TensorFlow autoencoder
	"""

	def __init__(self, model_idx, rom_domain, solver, sol_domain):

		if (rom_domain.time_integrator.time_type == "explicit"):
			raise ValueError("NLM LSPG with an explicit time integrator"
							+ "deteriorates to Galerkin, please use Galerkin"
							+ " or select an implicit time integrator.")

		if rom_domain.time_integrator.dual_time:
			raise ValueError("LSPG is intended for conservative variable"
							+ " evolution, please set dual_time = False")

		super().__init__(model_idx, rom_domain, solver, sol_domain)

		if self.encoder_jacob:
			raise ValueError("LSPG is not equipped with an encoder Jacobian"
							+ " approximation, please set encoder_jacob = False")

	def calcDCode(self, res_jacob, res, sol_domain):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		# decoder Jacobian, scaled
		jacob = self.calc_model_jacobian(sol_domain)
		scaled_jacob = jacob * self.norm_fac_prof_cons.ravel(order="C")[:, None]

		# test basis
		test_basis = (
			(res_jacob.toarray()
			/ self.norm_fac_prof_cons.ravel(order="C")[:, None])
			@ scaled_jacob
		)

		# Newton iteration linear solve
		lhs = test_basis.T @ test_basis
		rhs = (
			test_basis.T
			@ (res / self.norm_fac_prof_cons).ravel(order="C")
		)
		d_code = np.linalg.solve(lhs, rhs)

		return d_code, lhs, rhs

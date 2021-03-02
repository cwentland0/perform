from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import autoencoderTFKeras

import numpy as np
import tensorflow as tf


class autoencoderLSPGProjTFKeras(autoencoderTFKeras):
	"""
	Class for computing non-linear least-squares Petrov-Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if (romDomain.timeIntegrator.timeType == "explicit"):
			raise ValueError("NLM LSPG with an explicit time integrator deteriorates to Galerkin, please use Galerkin or select an implicit time integrator.")

		if romDomain.timeIntegrator.dualTime:
			raise ValueError("LSPG is intended for conservative variable evolution, please set dualTime = False")

		super().__init__(modelIdx, romDomain, solver, solDomain)

		if self.encoderJacob:
			raise ValueError("LSPG is not equipped with an encoder Jacobian approximation, please set encoderJacob = False")

	
	def calcDCode(self, resJacob, res, solDomain):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		# decoder Jacobian, scaled
		jacob = self.calcModelJacobian(solDomain)
		scaledJacob = jacob * self.normFacProfCons.ravel(order="C")[:,None]

		# test basis
		testBasis = (resJacob.toarray() / self.normFacProfCons.ravel(order="C")[:,None]) @ scaledJacob

		# Newton iteration linear solve
		LHS = testBasis.T @ testBasis
		RHS = testBasis.T @ (res / self.normFacProfCons).ravel(order="C")
		dCode = np.linalg.solve(LHS, RHS)
		
		return dCode, LHS, RHS
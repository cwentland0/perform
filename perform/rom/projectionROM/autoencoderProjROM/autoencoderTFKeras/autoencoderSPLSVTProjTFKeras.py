from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import autoencoderTFKeras

import numpy as np
import tensorflow as tf


class autoencoderSPLSVTProjTFKeras(autoencoderTFKeras):
	"""
	Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if (romDomain.timeIntegrator.timeType == "explicit"):
			raise ValueError("Explicit NLM SP-LSVT not implemented yet")

		if ((romDomain.timeIntegrator.timeType == "implicit") and (not romDomain.timeIntegrator.dualTime)):
			raise ValueError("NLM SP-LSVT is intended for primitive variable evolution, please use Galerkin or LSPG, or set dualTime = True")

		super().__init__(modelIdx, romDomain, solver, solDomain)

		if self.encoderJacob:
			raise ValueError("SP-LSVT is not equipped with an encoder Jacobian approximation, please set encoderJacob = False")


	def calcDCode(self, resJacob, res, solDomain):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		# decoder Jacobian, scaled
		jacob = self.calcModelJacobian(solDomain)
		scaledJacob = jacob * self.normFacProfPrim.ravel(order="C")[:,None]

		# test basis
		testBasis = (resJacob.toarray() / self.normFacProfCons.ravel(order="C")[:,None]) @ scaledJacob

		# Newton iteration linear solve
		LHS = testBasis.T @ testBasis
		RHS = testBasis.T @ (res / self.normFacProfCons).ravel(order="C")
		dCode = np.linalg.solve(LHS, RHS)
		
		return dCode, LHS, RHS
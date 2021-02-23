from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import autoencoderTFKeras

import numpy as np
import tensorflow as tf
from scipy.linalg import pinv


class autoencoderLSPGProjTFKeras(autoencoderTFKeras):
	"""
	Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if (romDomain.timeIntegrator.timeType == "explicit"):
			raise ValueError("NLM LSPG with an explicit time integrator deteriorates to Galerkin, please use Galerkin or select an implicit time integrator.")

		if romDomain.timeIntegrator.dualTime:
			raise ValueError("LSPG is intended for conservative variable evolution, please set dualTime = False")

		super().__init__(modelIdx, romDomain, solver, solDomain)

	
	def calcDCode(self, resJacob, res):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		if (self.encoderJacob):
			raise ValueError("Encoder Jacobian not implemented yet")

		else:

			if self.numericalJacob:
				jacob = self.calcNumericalTFJacobian(self.decoder, self.code)
			else:
				self.jacobInput.assign(self.code[None,:])
				jacobTF = self.calcAnalyticalModelJacobian(self.decoder, self.jacobInput)
				jacob = tf.squeeze(jacobTF, axis=[0,3]).numpy()

			# put in "channels-first" and flatten vars and cells dims
			if (self.ioFormat == "NHWC"):
				jacob = np.transpose(jacob, axes=(1,0,2))
			jacob = np.reshape(jacob, (-1, self.latentDim), order='C')

			scaledJacob = jacob * self.normFacProfCons.ravel(order="C")[:,None]

			testBasis = (resJacob.toarray() / self.normFacProfCons.ravel(order="C")[:,None]) @ scaledJacob

			# TODO: some of this may generalize to encoderJacob
			LHS = testBasis.T @ testBasis
			RHS = testBasis.T @ (res / self.normFacProfCons).ravel(order="C")

			# linear solve
			dCode = np.linalg.solve(LHS, RHS)
		
		return dCode, LHS, RHS
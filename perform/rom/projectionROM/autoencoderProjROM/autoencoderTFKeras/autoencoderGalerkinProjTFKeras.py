from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import autoencoderTFKeras

import numpy as np
import tensorflow as tf
from scipy.linalg import pinv


class autoencoderGalerkinProjTFKeras(autoencoderTFKeras):
	"""
	Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if (romDomain.timeIntegrator.timeType == "implicit"):
			raise ValueError("Implicit TF-Keras NLM Galerkin not implemented yet")

		super().__init__(modelIdx, romDomain, solver, solDomain)


	def calcProjector(self, solDomain):
		"""
		Compute RHS projection operator
		Decoder projector is pseudo-inverse of decoder Jacobian
		Encoder projector is just encoder Jacobian
		"""

		# TODO: could probably move some of this stuff into autoencoderTFKeras

		if self.encoderJacob:
			# TODO: only calculate the standardized solution once, hang onto it
			# 	don't have to pass solDomain, too
			sol = self.standardizeData(solDomain.solInt.solCons[self.varIdxs, :], 
										normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons, 
										center=True, centProf=self.centProfCons, inverse=False)
			
			self.jacobInput.assign(sol[None,:,:])
			jacobTF = self.calcAnalyticalModelJacobian(self.encoder, self.jacobInput)
			jacob = tf.squeeze(jacobTF, axis=[0,2]).numpy()

			if (self.ioFormat == "NHWC"):
				jacob = np.transpose(jacob, axes=(0,2,1))

			self.projector = np.reshape(jacob, (self.latentDim, -1), order='C')

		else:

			if self.numericalJacob:
				jacob = self.calcNumericalTFJacobian(self.decoder, self.code)
			else:
				self.jacobInput.assign(self.code[None,:])
				jacobTF = self.calcAnalyticalModelJacobian(self.decoder, self.jacobInput)
				jacob = tf.squeeze(jacobTF, axis=[0,3]).numpy()

			if (self.ioFormat == "NHWC"):
				jacob = np.transpose(jacob, axes=(1,0,2))
			jacob = np.reshape(jacob, (-1, self.latentDim), order='C')

			self.projector = pinv(jacob)


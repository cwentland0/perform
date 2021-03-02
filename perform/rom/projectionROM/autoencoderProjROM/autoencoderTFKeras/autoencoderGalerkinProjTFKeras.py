from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import autoencoderTFKeras

import numpy as np
import tensorflow as tf
from scipy.linalg import pinv


class autoencoderGalerkinProjTFKeras(autoencoderTFKeras):
	"""
	Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if romDomain.timeIntegrator.dualTime:
			raise ValueError("Galerkin is intended for conservative variable evolution, please set dualTime = False")

		super().__init__(modelIdx, romDomain, solver, solDomain)


	def calcProjector(self, solDomain):
		"""
		Compute RHS projection operator
		Decoder projector is pseudo-inverse of decoder Jacobian
		Encoder projector is just encoder Jacobian
		"""

		jacob = self.calcModelJacobian(solDomain)

		if self.encoderJacob:
			
			self.projector = jacob

		else:

			self.projector = pinv(jacob)


	def calcDCode(self, resJacob, res, solDomain):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		TODO: this is non-general and janky, only valid for BDF 
		"""

		jacob = self.calcModelJacobian(solDomain)

		if (self.encoderJacob):
			jacobPinv = jacob * self.normFacProfCons.ravel(order="C")[None,:]

		else:

			# decoder Jacobian, scaled
			scaledJacob = jacob * self.normFacProfCons.ravel(order="C")[:,None]
			jacobPinv = pinv(scaledJacob)

		# Newton iteration linear solve
		LHS = jacobPinv @ (resJacob.toarray() / self.normFacProfCons.ravel(order="C")[:,None]) @ scaledJacob
		RHS = jacobPinv @ (res / self.normFacProfCons).ravel(order="C")
		dCode = np.linalg.solve(LHS, RHS)

		return dCode, LHS, RHS
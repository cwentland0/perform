from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderTFKeras import autoencoderTFKeras

import numpy as np
import tensorflow as tf
from scipy.linalg import pinv


class autoencoderGalerkinProjTFKeras(autoencoderTFKeras):
	"""
	Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		super().__init__(modelIdx, romDomain, solver, solDomain)


	def calcProjector(self, solDomain):
		"""
		Compute RHS projection operator
		Decoder projector is pseudo-inverse of decoder Jacobian
		Encoder projector is just encoder Jacobian
		"""

		if (self.encoderJacob):
			# TODO: only calculate the standardized solution once, hang onto it
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

			self.jacobInput.assign(self.code[None,:])
			jacobTF = self.calcAnalyticalModelJacobian(self.decoder, self.jacobInput)
			jacob = tf.squeeze(jacobTF, axis=[0,3]).numpy()

			if (self.ioFormat == "NHWC"):
				jacob = np.transpose(jacob, axes=(1,0,2))
			jacob = np.reshape(jacob, (-1, self.latentDim), order='C')

			self.projector = pinv(jacob)



############

	# def calcNumericalTFJacobian(self, modelObj, encoder=False, dtype=realType):
	# 	"""
	# 	Compute numerical Jacobian of TensorFlow-Keras models by finite difference approximation
	# 	"""

	# 	if self.encoder:
	# 		code = evalEncoder(sol,u0,encoder,normData)
	# 		numJacob = np.zeros((code.shape[0],sol.shape[0]),dtype=np.float64)
	# 		sol = scaleOp(sol - u0, normData)
	# 		for elem in range(0,sol.shape[0]):
	# 			tempSol = sol.copy()
	# 			tempSol[elem] = tempSol[elem] + stepSize
	# 			output = np.squeeze(encoder.predict(np.array([tempSol,])))
	# 			numJacob[:,elem] = (output - code).T/stepSize/normData[1] 

	# 	else:
	# 		uSol = np.squeeze(decoder.predict(np.array([code,]))) 
	# 		numJacob = np.zeros((uSol.shape[0],code.shape[0]),dtype=np.float64)
	# 		for elem in range(0,code.shape[0]):
	# 			tempCode = code.copy()
	# 			tempCode[elem] = tempCode[elem] + stepSize 
	# 			output = np.squeeze(decoder.predict(np.array([tempCode,])))
	# 			numJacob[:,elem] = (output - uSol).T/stepSize

	# 	# def extractNumJacobian_test(decoder,code,uSol,u0,normData,stepSize):
	# 	# 	uSol = (uSol - u0 - normData[0])/normData[1]
	# 	# 	uzero = np.zeros(u0.shape) 
	# 	# 	normDataNoSub = np.array([0.0, 1.0])
	# 	# 	numJacob = np.zeros((uSol.shape[0],code.shape[0]),dtype=np.float64)
	# 	# 	for elem in range(0,code.shape[0]):
	# 	# 		tempCode = code.copy()
	# 	# 		tempCode[elem] = tempCode[elem] + stepSize 
	# 	# 		output = evalDecoder(tempCode,uzero,decoder,normDataNoSub)
	# 	# 		numJacob[:,elem] = (output - uSol).T/stepSize*normData[1]

	# 	return numJacob

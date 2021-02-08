from perform.constants import realType, fdStepDefault
from perform.inputFuncs import catchInput
from perform.rom.projectionROM.autoencoderProjROM.autoencoderProjROM import autoencoderProjROM

import numpy as np
from tensorflow.keras.models import load_model
from scipy.linalg import pinv

import pdb

class autoencoderGalerkinProjTFKeras(autoencoderProjROM):
	"""
	Model class for computing non-linear ROM's via TensorFlow autoencoder
	See user guide for expected format of encoder/decoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		super().__init__(modelIdx, romDomain, solver, solDomain)

		pass


	def loadModelObj(self, modelPath):
		"""
		Load decoder/encoder object from file specified by modelPath
		"""

		modelObj = load_model(modelPath, compile=False)
		return modelObj


	def getIOShape(self, shape):
		"""
		Handles situations in which layer shape is given as a list instead of a tuple
		"""

		if type(shape) is list:
			if (len(shape) != 1):
				raise ValueError("Invalid TF model I/O size")
			else:
				shape = shape[0]

		return shape


	def checkModelDims(self, decoder=True):
		"""
		Check decoder/encoder input/output dimensions
		"""

		if decoder:
			inputShape  = self.getIOShape(self.decoder.layers[0].input_shape)
			outputShape = self.getIOShape(self.decoder.layers[-1].output_shape)

			assert(inputShape[-1] == self.latentDim), "Mismatched decoder input shape: "+str(inputShape)+", "+str(self.latentDim)
			assert(outputShape[-2:] == self.solShape), "Mismatched decoder output shape: "+str(outputShape)+", "+str(self.solShape)

		else:
			inputShape  = self.getIOShape(self.encoder.layers[0].input_shape)
			outputShape = self.getIOShape(self.encoder.layers[-1].output_shape)

			assert(inputShape[-2:] == self.solShape), "Mismatched encoder output shape: "+str(inputShape)+", "+str(self.solShape)
			assert(outputShape[-1] == self.latentDim), "Mismatched encoder output shape: "+str(outputShape)+", "+str(self.latentDim)


	def applyDecoder(self, code):
		"""
		Compute raw decoding of code, without de-normalizing or de-centering
		"""

		sol = np.squeeze(self.decoder.predict(code[None,:]), axis=0)
		return sol
		

	def applyEncoder(self, sol):
		"""
		Compute raw encoding of solution, assuming it has been centered and normalized
		"""

		code = np.squeeze(self.encoder.predict(sol[None,:,:]), axis=0)
		return code


	def calcAnalyticalJacobian(inputArr, encoder=False):
		"""
		Compute analytical Jacobian of TensorFlow-Keras models using GradientTape
		For encoder, inputArr should be solution array
		For decoder, inputArr should be latent code
		"""

		if encoder:
			raise ValueError("Analytical encoder Jacobian has not been implemented")
		else:
			with tf.GradientTape() as g:
				inputs = tf.Variable(inputArr[None,:], dtype=dtype)
				outputs = self.decoder(inputs)

			# output of model is in CW order, Jacobian is thus CWK
			jacob = np.squeeze(g.jacobian(outputs, inputs).numpy(), axis=(0,3))

		return jacob


	def calcProjector(self, solDomain, runCalc=True):
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
			self.projector = self.calcAnalyticalJacobian(sol, encoder=True)
		else:
			jacob = self.calcAnalyticalJacobian(self.code, encoder=False)
			self.projector = pinv(jacob)


	# def calcNumericalTFJacobian(modelObj, encoder=False, dtype=realType):
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



	# # TODO: any way to put this in projectionROM?
	# def calcDCode(self, resJacob, res):
	# 	"""
	# 	Compute change in low-dimensional state for implicit scheme Newton iteration
	# 	"""

	# 	# calculate test basis
	# 	# TODO: this is not valid for scalar POD, another reason to switch to C ordering of resJacob
	# 	self.testBasis = (resJacob.toarray() / self.normFacProfCons.ravel(order="F")[:,None]) @ self.trialBasisFScaled

	# 	# compute W^T * W
	# 	LHS = self.testBasis.T @ self.testBasis
	# 	RHS = -self.testBasis.T @ res.ravel(order="F")

	# 	# linear solve
	# 	dCode = np.linalg.solve(LHS, RHS)
	# 	pdb.set_trace()
		
	# 	return dCode, LHS, RHS

from perform.constants import realType, fdStepDefault
from perform.inputFuncs import catchInput
from perform.rom.projectionROM.autoencoderProjROM.autoencoderProjROM import autoencoderProjROM

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.linalg import pinv

import pdb

# TODO: include casting to realType

class autoencoderGalerkinProjTFKeras(autoencoderProjROM):
	"""
	Model class for computing non-linear ROM's via TensorFlow autoencoder
	See user guide for expected format of encoder/decoder
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		self.runGPU = catchInput(romDomain.romDict, "runGPU", True)
		if self.runGPU:
			# make sure TF doesn't gobble up device memory
			gpus = tf.config.experimental.list_physical_devices('GPU')
			if gpus:
				try:
					# Currently, memory growth needs to be the same across GPUs
					for gpu in gpus:
						tf.config.experimental.set_memory_growth(gpu, True)
						logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				except RuntimeError as e:
					# Memory growth must be set before GPUs have been initialized
					print(e)
		else:
			# If GPU is available, TF will automatically run there
			#	This forces to run on CPU even if GPU is available
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

		self.ioFormat = romDomain.romDict["ioFormat"]
		if (self.ioFormat == "NCHW"):
			assert (self.runGPU), "Tensorflow cannot handle NCHW on CPUs"
		elif (self.ioFormat == "NHWC"):
			pass # works on GPU or CPU
		else:
			raise ValueError("ioFormat must be either \"NCHW\" or \"NHWC\"; you entered " + str(self.ioFormat))

		super().__init__(modelIdx, romDomain, solver, solDomain)

		# initialize tf.Variable for Jacobian calculations
		# otherwise, recreating this will cause retracing of the computational graph
		if (not self.numericalJacob):
			if self.encoderJacob:
				self.jacobInput = tf.Variable(solDomain.solInt.solCons[None,self.varIdxs,:], dtype=self.encoderIODtypes[0])
			else:
				self.jacobInput = tf.Variable(self.code[None,:], dtype=self.decoderIODtypes[0])


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


	def checkModel(self, decoder=True):
		"""
		Check decoder/encoder input/output dimensions and returns I/O dtypes
		"""

		if decoder:
			inputShape  = self.getIOShape(self.decoder.layers[0].input_shape)
			outputShape = self.getIOShape(self.decoder.layers[-1].output_shape)

			assert(inputShape[-1] == self.latentDim), "Mismatched decoder input shape: "+str(inputShape[-1])+", "+str(self.latentDim)
			if (self.ioFormat == "NCHW"):
				assert(outputShape[-2:] == self.solShape), "Mismatched decoder output shape: "+str(outputShape[-2:])+", "+str(self.solShape)
			else:
				assert(outputShape[-2:] == self.solShape[::-1]), "Mismatched decoder output shape: "+str(outputShape[-2:])+", "+str(self.solShape[::-1])

			inputDtype  = self.decoder.layers[0].dtype
			outputDtype = self.decoder.layers[-1].dtype

		else:
			inputShape  = self.getIOShape(self.encoder.layers[0].input_shape)
			outputShape = self.getIOShape(self.encoder.layers[-1].output_shape)

			assert(outputShape[-1] == self.latentDim), "Mismatched encoder output shape: "+str(outputShape[-1])+", "+str(self.latentDim)
			if (self.ioFormat == "NCHW"):
				assert(inputShape[-2:] == self.solShape), "Mismatched encoder output shape: "+str(inputShape[-2:])+", "+str(self.solShape)
			else:
				assert(inputShape[-2:] == self.solShape[::-1]), "Mismatched encoder output shape: "+str(inputShape[-2:])+", "+str(self.solShape[::-1])

			inputDtype  = self.encoder.layers[0].dtype
			outputDtype = self.encoder.layers[-1].dtype

		return [inputDtype, outputDtype]


	def applyDecoder(self, code):
		"""
		Compute raw decoding of code, without de-normalizing or de-centering
		"""

		sol = np.squeeze(self.decoder.predict(code[None,:]), axis=0)
		if (self.ioFormat == "NHWC"):
			sol = sol.T

		return sol
		

	def applyEncoder(self, sol):
		"""
		Compute raw encoding of solution, assuming it has been centered and normalized
		"""

		if (self.ioFormat == "NHWC"):
			solIn = (sol.copy()).T
		else:
			solIn = sol.copy()
		code = np.squeeze(self.encoder.predict(solIn[None,:,:]), axis=0)

		return code


	@tf.function
	def calcAnalyticalModelJacobian(self, model, inputs):
		"""
		Compute analytical Jacobian of TensorFlow-Keras model using GradientTape
		"""

		with tf.GradientTape() as g:
			outputs = model(inputs)
		jacob = g.jacobian(outputs, inputs)

		return jacob


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

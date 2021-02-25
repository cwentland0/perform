from perform.constants import realType, fdStepDefault
from perform.inputFuncs import catchInput
from perform.rom.projectionROM.autoencoderProjROM.autoencoderProjROM import autoencoderProjROM
from perform.rom.tfKerasFuncs import initDevice, loadModelObj, getIOShape

import tensorflow as tf
import numpy as np


class autoencoderTFKeras(autoencoderProjROM):
	"""
	Base class for any autoencoder projection-based ROMs using TensorFlow-Keras
	See user guide for notes on expected input format
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		self.runGPU = catchInput(romDomain.romDict, "runGPU", False)
		initDevice(self.runGPU)

		self.loadModelObj = loadModelObj # store function object for use in parent routines

		# "NCHW" (channels first) or "NHWC" (channels last)
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
				if self.targetCons:
					self.jacobInput = tf.Variable(solDomain.solInt.solCons[None,self.varIdxs,:], dtype=self.encoderIODtypes[0])
				else:
					self.jacobInput = tf.Variable(solDomain.solInt.solPrim[None,self.varIdxs,:], dtype=self.encoderIODtypes[0])
			else:
				self.jacobInput = tf.Variable(self.code[None,:], dtype=self.decoderIODtypes[0])


	def checkModel(self, decoder=True):
		"""
		Check decoder/encoder input/output dimensions and returns I/O dtypes
		"""

		if decoder:
			inputShape  = getIOShape(self.decoder.layers[0].input_shape)
			outputShape = getIOShape(self.decoder.layers[-1].output_shape)

			assert(inputShape[-1] == self.latentDim), "Mismatched decoder input shape: "+str(inputShape[-1])+", "+str(self.latentDim)
			if (self.ioFormat == "NCHW"):
				assert(outputShape[-2:] == self.solShape), "Mismatched decoder output shape: "+str(outputShape[-2:])+", "+str(self.solShape)
			else:
				assert(outputShape[-2:] == self.solShape[::-1]), "Mismatched decoder output shape: "+str(outputShape[-2:])+", "+str(self.solShape[::-1])

			inputDtype  = self.decoder.layers[0].dtype
			outputDtype = self.decoder.layers[-1].dtype

		else:
			inputShape  = getIOShape(self.encoder.layers[0].input_shape)
			outputShape = getIOShape(self.encoder.layers[-1].output_shape)

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

		sol = np.squeeze(self.decoder(code[None,:]).numpy(), axis=0)
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
		code = np.squeeze(self.encoder(solIn[None,:,:]).numpy(), axis=0)

		return code


	@tf.function
	def calcAnalyticalModelJacobian(self, model, inputs):
		"""
		Compute analytical Jacobian of TensorFlow-Keras model using GradientTape
		NOTE: inputs is a tf.Variable
		"""

		with tf.GradientTape() as g:
			outputs = model(inputs)
		jacob = g.jacobian(outputs, inputs)

		return jacob


	def calcNumericalModelJacobian(self, model, inputs):
		"""
		Compute numerical Jacobian of TensorFlow-Keras models by finite difference approximation
		NOTE: inputs is a np.ndarray
		"""

		if self.encoderJacob:
			raise ValueError("Numerical encoder Jacobian not implemented yet")
			if (self.ioFormat == "NHWC"):
				jacob = np.zeros((inputs.shape[0], self.numCells, self.numVars), dtype=realType)
			else:
				jacob = np.zeros((inputs.shape[0], self.numVars, self.numCells), dtype=realType)
		else:
			if (self.ioFormat == "NHWC"):
				jacob = np.zeros((self.numCells, self.numVars, inputs.shape[0]), dtype=realType)
			else:
				jacob = np.zeros((self.numVars, self.numCells, inputs.shape[0]), dtype=realType)

		# get initial prediction
		outputsBase = np.squeeze(model(inputs[None,:]).numpy(), axis=0)

		# TODO: this does not generalize to encoder Jacobian
		for elemIdx in range(0, inputs.shape[0]):

			# perturb
			inputsPert = inputs.copy()
			inputsPert[elemIdx] = inputsPert[elemIdx] + self.fdStep

			# make prediction at perturbed state
			outputs = np.squeeze(model(inputsPert[None,:]).numpy(), axis=0)

			# compute finite difference approximation
			jacob[:,:,elemIdx] = (outputs - outputsBase) / self.fdStep

		return jacob


	def calcModelJacobian(self):
		"""
		Helper function for calculating TensorFlow-Keras model Jacobian
		"""

		# TODO: generalize this for generic model, input

		if self.encoderJacob:
			# TODO: only calculate the standardized solution once, hang onto it
			# 	don't have to pass solDomain, too
			sol = self.standardizeData(solDomain.solInt.solCons[self.varIdxs, :],
										normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons,
										center=True, centProf=self.centProfCons, inverse=False)

			if self.numericalJacob:
				jacob = self.calcNumericalTFJacobian(self.encoder, sol)
			else:
				self.jacobInput.assign(sol[None,:,:])
				jacobTF = self.calcAnalyticalModelJacobian(self.encoder, self.jacobInput)
				jacob = tf.squeeze(jacobTF, axis=[0,2]).numpy()

			if (self.ioFormat == "NHWC"):
				jacob = np.transpose(jacob, axes=(0,2,1))

			jacob = np.reshape(jacob, (self.latentDim, -1), order='C')

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

		return jacob
from perform.constants import realType, fdStepDefault
from perform.inputFuncs import catchInput
from perform.rom.linearProjROM.linearProjROM import linearProjROM

import tensorflow as tf

class autoencoderGalerkinProjTF(linearProjROM):
	"""
	Model class for computing non-linear ROM's via TensorFlow autoencoder
	""""

	def __init__(self, romDict):

		
		self.numericalJacob = catchInput(romDict, "numericalJacob", False)
		if self.numericalJacob: self.fdStep = catchInput(romDict, "fdStep", fdStepDefault)

		self.encoder = catchInput(romDict, "encoderROM", False)

		# load models

		# check model dimensions against solDomain

		# do a test run just to make sure everything works okay

		pass



	def calcProjection(self):



	def calcAnalyticalTFJacobian(modelObj, dtype=realType):
		"""
		Compute analytical Jacobian of TensorFlow-Keras models using GradientTape
		"""

		if self.encoder:
			raise ValueError("Analytical encoder Jacobian has not been implemented")
		else:
			with tf.GradientTape() as g:
				inputs = tf.Variable(modelObj.code[None,:], dtype=dtype)
				outputs = modelObj.decoder(inputs)

			# output of model is in CW order, Jacobian is thus CWK, reorder to WCK 
			modelObj.modelJacob = np.transpose(np.squeeze(g.jacobian(outputs, inputs).numpy(), axis=(0,3)), axes=(1,0,2))


	def calcNumericalTFJacobian(modelObj, encoder=False, dtype=realType):
		"""
		Compute numerical Jacobian of TensorFlow-Keras models by finite difference approximation
		"""

		if self.encoder:
			code = evalEncoder(sol,u0,encoder,normData)
			numJacob = np.zeros((code.shape[0],sol.shape[0]),dtype=np.float64)
			sol = scaleOp(sol - u0, normData)
			for elem in range(0,sol.shape[0]):
				tempSol = sol.copy()
				tempSol[elem] = tempSol[elem] + stepSize
				output = np.squeeze(encoder.predict(np.array([tempSol,])))
				numJacob[:,elem] = (output - code).T/stepSize/normData[1] 

		else:
			uSol = np.squeeze(decoder.predict(np.array([code,]))) 
			numJacob = np.zeros((uSol.shape[0],code.shape[0]),dtype=np.float64)
			for elem in range(0,code.shape[0]):
				tempCode = code.copy()
				tempCode[elem] = tempCode[elem] + stepSize 
				output = np.squeeze(decoder.predict(np.array([tempCode,])))
				numJacob[:,elem] = (output - uSol).T/stepSize

		# def extractNumJacobian_test(decoder,code,uSol,u0,normData,stepSize):
		# 	uSol = (uSol - u0 - normData[0])/normData[1]
		# 	uzero = np.zeros(u0.shape) 
		# 	normDataNoSub = np.array([0.0, 1.0])
		# 	numJacob = np.zeros((uSol.shape[0],code.shape[0]),dtype=np.float64)
		# 	for elem in range(0,code.shape[0]):
		# 		tempCode = code.copy()
		# 		tempCode[elem] = tempCode[elem] + stepSize 
		# 		output = evalDecoder(tempCode,uzero,decoder,normDataNoSub)
		# 		numJacob[:,elem] = (output - uSol).T/stepSize*normData[1]

		return numJacob
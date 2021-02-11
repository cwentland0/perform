from perform.rom.projectionROM.projectionROM import projectionROM
from perform.inputFuncs import catchInput

import numpy as np
import os
import pdb

class autoencoderProjROM(projectionROM):

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		super().__init__(modelIdx, romDomain, solver)

		romDict = romDomain.romDict

		# load decoder
		decoderPath = os.path.join(romDomain.modelDir, romDomain.modelFiles[modelIdx])
		assert(os.path.isfile(decoderPath)), "Invalid decoder file path"
		self.decoder = self.loadModelObj(decoderPath)
		self.decoderIODtypes = self.checkModel(decoder=True)

		# if required, load encoder
		# encoder is required for encoder Jacobian or initializing from projection of full ICs
		self.encoderJacob = catchInput(romDict, "encoderJacob", False)
		self.encoder = None
		if (self.encoderJacob or (not romDomain.initROMFromFile[modelIdx])):
			encoderFiles = romDict["encoderFiles"]
			assert (len(encoderFiles) == romDomain.numModels), "Must provide encoderFiles for each model"
			encoderPath = os.path.join(romDomain.modelDir, encoderFiles[modelIdx])
			assert (os.path.isfile(encoderPath)), "Could not find encoder file at " + encoderPath
			self.encoder = self.loadModelObj(encoderPath)
			self.encoderIODtypes = self.checkModel(decoder=False)

		# numerical Jacobian params
		self.numericalJacob = catchInput(romDict, "numericalJacob", False)
		if self.numericalJacob: self.fdStep = catchInput(romDict, "fdStep", fdStepDefault)


	def encodeSol(self, solIn):
		"""
		Compute encoding of full-dimensional state, including centering and normalization
		"""

		if (self.targetCons):
			sol = self.standardizeData(solIn, 
										normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons, 
										center=True, centProf=self.centProfCons, inverse=False)
		else:
			sol = self.standardizeData(solIn, 
										normalize=True, normFacProf=self.normFacProfPrim, normSubProf=self.normSubProfPrim, 
										center=True, centProf=self.centProfPrim, inverse=False)

		code = self.applyEncoder(sol)

		return code


	def initFromSol(self, solDomain):
		"""
		Initialize full-order solution from projection of loaded full-order initial conditions
		"""

		if (self.targetCons):
			self.code = self.encodeSol(solDomain.solInt.solCons[self.varIdxs, :])
			solDomain.solInt.solCons[self.varIdxs, :] = self.decodeSol(self.code)
		else:
			self.code = self.encodeSol(solDomain.solInt.solPrim[self.varIdxs, :])
			solDomain.solInt.solPrim[self.varIdxs, :] = self.decodeSol(self.code)

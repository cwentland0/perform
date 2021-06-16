
from perform.inputFuncs import readInputFile, catchList, catchInput
from perform.rom.linearProjROM.linearGalerkinProj import linearGalerkinProj
from perform.timeIntegrator import getTimeIntegrator
from perform.solution.solutionPhys import solutionPhys
from perform.spaceSchemes import calcRHS
from perform.Jacobians import calcDResDSolPrim
import perform.constants as const

import numpy as np
from time import sleep
import os
from scipy.sparse import block_diag, csr_matrix
import pdb
import matplotlib.pyplot as plt

# TODO: when moving to multi-domain, it may be useful to just hold a solDomain inside romDomain for the associated full-dim solution
# 		Still a pain to move around since it's associated with the romDomain and not the romModel, but whatever

# TODO: I'm making this too general, I think it's safe for now to assume that a single domain will have a single time integrator,
#		single ROM method, single gas model, etc.

class romDomain:
	"""
	Container class for ROM parameters and romModels
	"""
	def __init__(self, solDomain, solver):

		romDict = readInputFile(solver.romInputs)
		self.romDict = romDict
		# load model parameters
		self.romMethod 		= str(romDict["romMethod"])
		self.numModels 		= int(romDict["numModels"])
		self.latentDims 	= catchList(romDict, "latentDims", [0], lenHighest=self.numModels)
		modelVarIdxs 		= catchList(romDict, "modelVarIdxs", [[-1]], lenHighest=self.numModels)

		# check model parameters
		for i in self.latentDims: assert (i > 0), "latentDims must contain positive integers"
		if (self.numModels == 1):
			assert (len(self.latentDims) == 1), "Must provide only one value of latentDims when numModels = 1"
			assert (self.latentDims[0] > 0), "latentDims must contain positive integers"
		else:
			if (len(self.latentDims) == self.numModels):
				pass
			elif (len(self.latentDims) == 1):	
				print("Only one value provided in latentDims, applying to all models")
				sleep(1.0)
				self.latentDims = [self.latentDims[0]] * self.numModels
			else:
				raise ValueError("Must provide either numModels or 1 entry in latentDims")

		# load and check modelVarIdxs
		for modelIdx in range(self.numModels):
			assert (modelVarIdxs[modelIdx][0] != -1), "modelVarIdxs input incorrectly, probably too few lists"
		assert (len(modelVarIdxs) == self.numModels), "Must specify modelVarIdxs for every model"
		modelVarSum = 0
		for modelIdx in range(self.numModels):
			modelVarSum += len(modelVarIdxs[modelIdx])
			for modelVarIdx in modelVarIdxs[modelIdx]:
				assert (modelVarIdx >= 0), "modelVarIdxs must be non-negative integers"
				assert (modelVarIdx < solDomain.gasModel.numEqs), "modelVarIdxs must less than the number of governing equations"
		assert (modelVarSum == solDomain.gasModel.numEqs), ("Must specify as many modelVarIdxs entries as governing equations (" +
														str(modelVarSum) + " != " + str(solDomain.gasModel.numEqs) + ")")
		modelVarIdxsOneList = sum(modelVarIdxs, [])
		assert (len(modelVarIdxsOneList) == len(set(modelVarIdxsOneList))), "All entries in modelVarIdxs must be unique"
		self.modelVarIdxs = modelVarIdxs 

		# load and check model input locations
		self.modelDir 	= str(romDict["modelDir"])
		modelFiles = romDict["modelFiles"]
		self.modelFiles = [None] * self.numModels
		assert (len(modelFiles) == self.numModels), "Must provide modelFiles for each model"
		for modelIdx in range(self.numModels):
			inFile = os.path.join(self.modelDir, modelFiles[modelIdx])
			assert (os.path.isfile(inFile)), "Could not find model file at " + inFile
			self.modelFiles[modelIdx] = inFile

		# load standardization profiles, if they are required
		self.normSubConsIn 	= catchList(romDict, "normSubConsIn", [""])
		self.normFacConsIn 	= catchList(romDict, "normFacConsIn", [""])
		self.centConsIn 	= catchList(romDict, "centConsIn", [""])
		self.normSubPrimIn 	= catchList(romDict, "normSubPrimIn", [""])
		self.normFacPrimIn 	= catchList(romDict, "normFacPrimIn", [""])
		self.centPrimIn 	= catchList(romDict, "centPrimIn", [""])

		# load low-dimensional initial condition state, if desired
		self.loadInitCode(romDict)

		self.setModelFlags()

		# Adaptive ROM
		self.adaptiveROM = catchInput(romDict, "adaptiveROM", False)
		if self.adaptiveROM:
				self.adaptiveROMMethod 				=	catchInput(romDict, "adaptiveROMMethod", "OSAB")
				self.adaptROMResidualSampStep		=	catchInput(romDict, "adaptROMResidualSampStep", 1)
				self.adaptROMnumResSamp				=	catchInput(romDict, "adaptROMnumResSamp", solver.mesh.numCells)
				self.adaptROMInitialSnap			=	catchInput(romDict, "adaptROMInitialSnap", 1)
				self.adaptROMWindowSize				=	catchInput(romDict, "adaptROMWindowSize", 1)
				self.adaptROMUpdateRank				=	catchInput(romDict, "adaptROMUpdateRank", 1)
				self.adaptROMConsSnapFName			=	catchInput(romDict, "adaptROMConsSnapFName", "solCons_FOM.npy")


		# set up hyper-reduction, if necessary
		self.hyperReduc = catchInput(romDict, "hyperReduc", False)
		if (self.isIntrusive and self.hyperReduc):
			self.loadHyperReduc(solDomain, solver)

		# initialize models for domain
		self.modelList = [None] * self.numModels
		for modelIdx in range(self.numModels):

			if (self.romMethod == "linearGalerkinProj"):
				self.modelList[modelIdx] = linearGalerkinProj(modelIdx, self, solver, solDomain)
			elif (self.romMethod == "linearLSPGProj"):
				raise ValueError("linearLSPGProj ROM not implemented yet")
			elif (self.romMethod == "linearSPLSVTProj"):
				raise ValueError("linearSPLSVTProj ROM not implemented yet")
			elif (self.romMethod == "autoencoderGalerkinProjTF"):
				raise ValueError("autoencoderGalerkinProjTF ROM not implemented yet")
			elif (self.romMethod == "autoencoderLSPGProjTF"):
				raise ValueError("autoencoderLSPGProjTF ROM not implemented yet")
			elif (self.romMethod == "autoencoderSPLSVTProjTF"):
				raise ValueError("autoencoderSPLSVTProjTF ROM not implemented yet")
			elif (self.romMethod == "liftAndLearn"):
				raise ValueError("liftAndLearn ROM not implemented yet")
			elif (self.romMethod == "tcnNonintrusive"):
				raise ValueError("tcnNonintrusive ROM not implemented yet")
			else:
				raise ValueError("Invalid ROM method name: "+self.romMethod)

			# initialize state
			if self.initROMFromFile[modelIdx]:
				self.modelList[modelIdx].initFromCode(self.code0[modelIdx], solDomain, solver)
			else:
				self.modelList[modelIdx].initFromSol(solDomain, solver)

		solDomain.solInt.updateState(fromCons=self.targetCons)

		# get time integrator, if necessary
		# TODO: timeScheme should be specific to the romDomain, not the solver
		if self.hasTimeIntegrator:
			self.timeIntegrator = getTimeIntegrator(solver.timeScheme, solver.paramDict)

			# initialize code history
			# TODO: this is necessary for non-time-integrated methods, e.g. TCN
			for model in self.modelList:
				model.codeHist = [model.code.copy()] * (self.timeIntegrator.timeOrder+1)

		else:
			self.timeIntegrator = None 	# TODO: this might be pointless

		# overwrite history with initialized solution
		solDomain.solInt.solHistCons = [solDomain.solInt.solCons.copy()] * (self.timeIntegrator.timeOrder+1)
		solDomain.solInt.solHistPrim = [solDomain.solInt.solPrim.copy()] * (self.timeIntegrator.timeOrder+1)


		# gather solution history from stale solution data
		if self.adaptiveROM and self.adaptiveROMMethod=='AADEIM':
			assert (os.path.exists(os.path.join(const.unsteadyOutputDir, self.adaptROMConsSnapFName))), \
				'Path for stale conservative state : '+ str(os.path.join(const.unsteadyOutputDir, self.adaptROMConsSnapFName))+ ' does not exist'

			self.staleConsSnapshots = np.load(os.path.join(const.unsteadyOutputDir, self.adaptROMConsSnapFName))
			assert (self.staleConsSnapshots.shape[1] == solver.mesh.numCells), 'resolution mis-match between stale states and current simulation'

			self.timeIntegrator.staleStatetimeOrder = min(self.timeIntegrator.timeOrder,self.staleConsSnapshots.shape[-1])
			solDomain.timeIntegrator.staleStatetimeOrder = self.timeIntegrator.staleStatetimeOrder

			for timeIdx in range(1, self.timeIntegrator.staleStatetimeOrder + 1):
				solDomain.solInt.solHistCons[timeIdx] = self.staleConsSnapshots[:, :, -timeIdx].copy()

			solDomain.solInt.solHistCons[0] = solDomain.solInt.solHistCons[1].copy()
			solDomain.solInt.solCons = solDomain.solInt.solHistCons[0].copy()
			solDomain.solInt.updateState(fromCons=True)

			for modelIdx, model in enumerate(self.modelList): model.adapt.initializeLookBackWindow(self, model)


	def setModelFlags(self):
		"""
		Set universal ROM method flags that dictate various execution behaviors
		"""

		self.hasTimeIntegrator 	= False 	# whether the model uses numerical time integration to advance the solution
		self.isIntrusive 		= False 	# whether the model requires access to the ODE RHS calculation
		self.targetCons 		= False 	# whether the ROM models the conservative variables
		self.targetPrim 		= False 	# whether the ROM models the primitive variables

		self.hasConsNorm 		= False 	# whether the model must load conservative variable normalization profiles
		self.hasConsCent 		= False 	# whether the model must load conservative variable centering profile
		self.hasPrimNorm 		= False		# whether the model must load primitive variable normalization profiles
		self.hasPrimCent 		= False		# whether the model must load primitive variable centering profile

		if (self.romMethod == "linearGalerkinProj"):
			self.hasTimeIntegrator 	= True
			self.isIntrusive 	   	= True
			self.targetCons 		= True
			self.hasConsNorm 		= True
			self.hasConsCent 		= True
		elif (self.romMethod == "linearLSPGProj"):
			self.hasTimeIntegrator 	= True
			self.isIntrusive 	   	= True
			self.targetCons 		= True
			self.hasConsNorm 		= True
			self.hasConsCent 		= True
		elif (self.romMethod == "linearSPLSVTProj"):
			self.hasTimeIntegrator 	= True
			self.isIntrusive 	   	= True
			self.targetPrim 		= True
			self.hasConsNorm 		= True
			self.hasPrimNorm 		= True
			self.hasPrimCent 		= True
		elif (self.romMethod == "autoencoderGalerkinProjTF"):
			self.hasTimeIntegrator 	= True
			self.isIntrusive 	   	= True
			self.targetCons 		= True
			self.hasConsNorm 		= True
			self.hasConsCent 		= True
		elif (self.romMethod == "autoencoderLSPGProjTF"):
			self.hasTimeIntegrator 	= True
			self.isIntrusive 	   	= True
			self.targetCons 		= True
			self.hasConsNorm 		= True
			self.hasConsCent 		= True
		elif (self.romMethod == "autoencoderSPLSVTProjTF"):
			self.hasTimeIntegrator 	= True
			self.isIntrusive 	   	= True
			self.targetPrim 		= True
			self.hasConsNorm 		= True
			self.hasPrimNorm 		= True
			self.hasPrimCent 		= True
		elif (self.romMethod == "liftAndLearn"):
			# TODO: the cons/prim dichotomy doesn't work for lifted variables
			self.hasTimeIntegrator 	= True
			raise ValueError("Finalize settings of method parameters for lift and learn")
		elif (self.romMethod == "tcnNonintrusive"):
			# TODO: TCN does not **need** to target one or the other
			self.targetCons = catchInput(self.romDict, "targetCons", False)
			self.targetPrim = catchInput(self.romDict, "targetPrim", False)
		else:
			raise ValueError("Invalid ROM method name: "+self.romMethod)

		# TODO: not strictly true for the non-intrusive models
		assert (self.targetCons != self.targetPrim), "Model must target either the primitive or conservative variables"


	def loadInitCode(self, romDict):
		"""
		Load low-dimensional state from disk if provided, overwriting full-dimensional state
		If not provided, ROM model will attempt to compute initial low-dimensional state from initial full-dimensional state
		"""

		self.lowDimInitFiles = catchList(romDict, "lowDimInitFiles", [""])
		
		lenInitFiles = len(self.lowDimInitFiles)
		self.initROMFromFile = [False] * self.numModels
		self.code0 = [None] * self.numModels

		# no init files given
		if ((lenInitFiles == 1) and (self.lowDimInitFiles[0] == "")):
			pass

		# TODO: this format leads to some messy storage, figure out a better way. Maybe .npz files?
		else:
			assert (lenInitFiles == self.numModels), ("If initializing any ROM model from a file, must provide list entries for every model. " +
													"If you don't wish to initialize from file for a model, input an empty string \"\" in the list entry.")
			for modelIdx in range(self.numModels):
				initFile = self.lowDimInitFiles[modelIdx]
				if (initFile != ""):
					assert (os.path.isfile(initFile)), "Could not find ROM initialization file at "+initFile
					self.code0[modelIdx] = np.load(initFile)
					self.initROMFromFile[modelIdx] = True


	def loadHyperReduc(self, solDomain, solver):
		"""
		Loads direct sampling indices and determines cell indices for calculating fluxes and gradients
		"""

		raise ValueError("Hyper-reduction temporarily out of commission for development")

		# TODO: add some explanations for what each index array accomplishes

		# load and check sample points
		sampFile = catchInput(self.romDict, "sampFile", "")
		assert (sampFile != ""), "Must supply sampFile if performing hyper-reduction"
		sampFile = os.path.join(self.modelDir, sampFile)
		assert (os.path.isfile(sampFile)), ("Could not find sampFile at " + sampFile)

		# NOTE: assumed that sample indices are zero-indexed
		solDomain.directSampIdxs = np.load(sampFile).flatten()
		solDomain.directSampIdxs = (np.sort(solDomain.directSampIdxs)).astype(np.int32)
		solDomain.numSampCells = len(solDomain.directSampIdxs)
		assert (solDomain.numSampCells <= solver.mesh.numCells), "Cannot supply more sampling points than cells in domain."
		assert (np.amin(solDomain.directSampIdxs) >= 0), "Sampling indices must be non-negative integers"
		assert (np.amax(solDomain.directSampIdxs) < solver.mesh.numCells), "Sampling indices must be less than the number of cells in the domain"
		assert (len(np.unique(solDomain.directSampIdxs)) == solDomain.numSampCells), "Sampling indices must be unique"

		# TODO: should probably shunt these over to a function for when indices get updated in adaptive method

		# compute indices for inviscid flux calculations
		# NOTE: have to account for fact that boundary cells are prepended/appended
		solDomain.fluxSampLIdxs = np.zeros(2 * solDomain.numSampCells, dtype=np.int32)
		solDomain.fluxSampLIdxs[0::2] = solDomain.directSampIdxs
		solDomain.fluxSampLIdxs[1::2] 	= solDomain.directSampIdxs + 1

		solDomain.fluxSampRIdxs = np.zeros(2 * solDomain.numSampCells, dtype=np.int32)
		solDomain.fluxSampRIdxs[0::2] = solDomain.directSampIdxs + 1
		solDomain.fluxSampRIdxs[1::2] 	= solDomain.directSampIdxs + 2

		# eliminate repeated indices
		solDomain.fluxSampLIdxs = np.unique(solDomain.fluxSampLIdxs)
		solDomain.fluxSampRIdxs = np.unique(solDomain.fluxSampRIdxs)
		solDomain.numFluxFaces  = len(solDomain.fluxSampLIdxs)

		# Roe average
		if (solver.spaceScheme == "roe"):
			zerosProf        = np.zeros((solDomain.gasModel.numEqs, solDomain.numFluxFaces), dtype=const.realType)
			solDomain.solAve = solutionPhys(solDomain, zerosProf, zerosProf, solDomain.numFluxFaces, solver)

		# to slice flux when calculating RHS
		solDomain.fluxRHSIdxs = np.zeros(solDomain.numSampCells, np.int32)
		for i in range(1,solDomain.numSampCells):
			if (solDomain.directSampIdxs[i] == (solDomain.directSampIdxs[i-1]+1)):
				solDomain.fluxRHSIdxs[i] = solDomain.fluxRHSIdxs[i-1] + 1
			else:
				solDomain.fluxRHSIdxs[i] = solDomain.fluxRHSIdxs[i-1] + 2

		# compute indices for gradient calculations
		# NOTE: also need to account for prepended/appended boundary cells
		# TODO: generalize for higher-order schemes
		if (solver.spaceOrder > 1):
			if (solver.spaceOrder == 2):
				solDomain.gradIdxs = np.concatenate((solDomain.directSampIdxs+1, solDomain.directSampIdxs, solDomain.directSampIdxs+2))
				solDomain.gradIdxs = np.unique(solDomain.gradIdxs)
				# exclude left neighbor of inlet, right neighbor of outlet
				if (solDomain.gradIdxs[0] == 0): solDomain.gradIdxs = solDomain.gradIdxs[1:]
				if (solDomain.gradIdxs[-1] == (solver.mesh.numCells+1)):  solDomain.gradIdxs = solDomain.gradIdxs[:-1]
				solDomain.numGradCells = len(solDomain.gradIdxs)

				# neighbors of gradient cells
				solDomain.gradNeighIdxs = np.concatenate((solDomain.gradIdxs-1, solDomain.gradIdxs+1))
				solDomain.gradNeighIdxs = np.unique(solDomain.gradNeighIdxs)
				# exclude left neighbor of inlet, right neighbor of outlet
				if (solDomain.gradNeighIdxs[0] == -1): solDomain.gradNeighIdxs = solDomain.gradNeighIdxs[1:]
				if (solDomain.gradNeighIdxs[-1] == (solver.mesh.numCells+2)):  solDomain.gradNeighIdxs = solDomain.gradNeighIdxs[:-1]

				# indices of gradIdxs in gradNeighIdxs
				_, _, solDomain.gradNeighExtract = np.intersect1d(solDomain.gradIdxs, solDomain.gradNeighIdxs, return_indices=True)

				# indices of gradIdxs in fluxSampLIdxs and fluxSampRIdxs, and vice versa
				_, solDomain.gradLExtract, solDomain.fluxLExtract = np.intersect1d(solDomain.gradIdxs, solDomain.fluxSampLIdxs, return_indices=True)
				_, solDomain.gradRExtract, solDomain.fluxRExtract = np.intersect1d(solDomain.gradIdxs, solDomain.fluxSampRIdxs, return_indices=True)

			else:
				raise ValueError("Sampling for higher-order schemes not implemented yet")

		# copy indices for ease of use
		self.numSampCells = solDomain.numSampCells
		self.directSampIdxs = solDomain.directSampIdxs

		# paths to hyper-reduction files (unpacked later)
		hyperReducFiles = self.romDict["hyperReducFiles"]
		self.hyperReducFiles = [None] * self.numModels
		assert (len(hyperReducFiles) == self.numModels), "Must provide hyperReducFiles for each model"
		for modelIdx in range(self.numModels):
			inFile = os.path.join(self.modelDir, hyperReducFiles[modelIdx])
			assert (os.path.isfile(inFile)), "Could not find hyper-reduction file at " + inFile
			self.hyperReducFiles[modelIdx] = inFile

		# load hyper reduction dimensions and check validity
		self.hyperReducDims = catchList(self.romDict, "hyperReducDims", [0], lenHighest=self.numModels)
		for i in self.hyperReducDims: assert (i > 0), "hyperReducDims must contain positive integers"
		if (self.numModels == 1):
			assert (len(self.hyperReducDims) == 1), "Must provide only one value of hyperReducDims when numModels = 1"
			assert (self.hyperReducDims[0] > 0), "hyperReducDims must contain positive integers"
		else:
			if (len(self.hyperReducDims) == self.numModels):
				pass
			elif (len(self.hyperReducDims) == 1):	
				print("Only one value provided in hyperReducDims, applying to all models")
				sleep(1.0)
				self.hyperReducDims = [self.hyperReducDims[0]] * self.numModels
			else:
				raise ValueError("Must provide either numModels or 1 entry in hyperReducDims")


	def advanceIter(self, solDomain, solver):
		"""
		Advance low-dimensional state forward one time iteration
		"""

		print("Iteration "+str(solver.iter))
		# update model which does NOT require numerical time integration
		if not self.hasTimeIntegrator:
			raise ValueError("Iteration advance for models without numerical time integration not yet implemented")

		# if method requires numerical time integration
		else:
			self.CombinedBasis = block_diag([model.trialBasis for modelIdx, model in enumerate(self.modelList)])
			if self.hyperReduc:
				raise ValueError('still in the works')
				self.sampledCombinedBasisPinv = block_diag(
					[np.linalg.pinv(model.trialBasis[solDomain.directSampIdxs, :]) for modelIdx, model in
					 enumerate(self.modelList)])

			if self.adaptiveROM and self.adaptiveROMMethod == 'AADEIM':
				if solDomain.timeIntegrator.dualTime: raise ValueError('AADEIM currently implemented for conservative variables...')
				for modelIdx, model in enumerate(self.modelList): model.adapt.initializeHistory(self, solDomain, solver, model)
				solDomain.solInt.updateState(fromCons=True)

			for self.timeIntegrator.subiter in range(self.timeIntegrator.subiterMax):
				self.advanceSubiter(solDomain, solver)

				if (self.timeIntegrator.timeType == "implicit"):
					self.calcCodeResNorms(solDomain, solver, self.timeIntegrator.subiter)
					if (solDomain.solInt.resNormL2 < self.timeIntegrator.resTol):
						break

			if self.adaptiveROM and self.adaptiveROMMethod == 'AADEIM':
				#There exists a bottle-neck : Basis update pools all the residual sampling points.
				# In the 'model' structure, two different loops are required to first gather the sampling points and then update the basis.

				self.residualCombinedSampleIdx = []
				for modelIdx, model in enumerate(self.modelList):model.adapt.gatherSamplePoints(self, solDomain, solver, model)
				for modelIdx, model in enumerate(self.modelList): model.adapt.adaptModel(self, solDomain, solver, model)

		solDomain.solInt.updateSolHist()
		self.updateCodeHist()


	def advanceSubiter(self, solDomain, solver):
		"""
		Advance physical solution forward one subiteration of time integrator
		"""
		solInt = solDomain.solInt
		solInt.res, resJacob = None, None

		# compute RHS if required
		if self.isIntrusive:
			calcRHS(solDomain, solver)

		# implicit time integrator
		if (self.timeIntegrator.timeType == "implicit"):
			if self.adaptiveROM and self.adaptiveROMMethod == 'OSAB': raise ValueError('OSAB not implemented for implicit framework')
			# compute residual and residual Jacobian, if required
			if self.isIntrusive:
				if solDomain.timeIntegrator.dualTime: raise ValueError('under construction')
				resVec = self.timeIntegrator.calcReducedResidualVec(self, solDomain, solver)
				resJacob = calcDResDSolPrim(solDomain, solver)

			scale = np.zeros((solDomain.gasModel.numEqs, solDomain.solInt.numCells))
			for modelIdx, model in enumerate(self.modelList): scale[model.varIdxs, :] = model.normFacProfCons

			LHS = self.CombinedBasis.T @ (resJacob / scale.ravel(order="C")[:, None]) @ (self.CombinedBasis.toarray() * scale.ravel(order = 'C')[:, None])
			RHS = resVec.reshape(-1)

			dCodeVec = np.linalg.solve(LHS, RHS)

			idx = np.insert(np.cumsum(self.latentDims), 0, 0)
			for modelIdx, model in enumerate(self.modelList):
				model.code = model.code+dCodeVec[idx[modelIdx]:idx[modelIdx+1]]
				model.codeHist[0] = model.code.copy()
				model.updateSol(solDomain)
				model.res = (LHS@dCodeVec[:, None] - RHS[:, None])[idx[modelIdx]:idx[modelIdx+1]]

			solInt.updateState(fromCons=True)
			solInt.solHistCons[0] = solInt.solCons.copy()
			solInt.solHistPrim[0] = solInt.solPrim.copy()

			if self.isIntrusive:
				calcRHS(solDomain, solver)

		# explicit time integrator
		else:

			for modelIdx, model in enumerate(self.modelList):
				if(self.adaptiveROM and model.adapt.adaptsubIteration):
					model.adaptSubIteration(self, solDomain)
				else:
					model.calcRHSLowDim(self, solDomain)
					dCode = self.timeIntegrator.solveSolChange(model.rhsLowDim)
					model.code = model.codeHist[0] + dCode
					model.updateSol(solDomain)

			solInt.updateState(fromCons=True)
			if self.isIntrusive: calcRHS(solDomain, solver)


	def updateCodeHist(self):
		"""
		Update low-dimensional state history after physical time step
		"""

		for model in self.modelList:

			model.codeHist[1:] = model.codeHist[:-1]
			model.codeHist[0]  = model.code.copy()

	def calcCodeResNorms(self, solDomain, solver, subiter):
		"""
		Calculate and print linear solve residual norms		
		Note that output is ORDER OF MAGNITUDE of residual norm (i.e. 1e-X, where X is the order of magnitude)
		"""

		# compute residual norm for each model
		normL2Sum = 0.0
		normL1Sum = 0.0
		for model in self.modelList:

			normL2, normL1 = model.calcCodeNorms()

			normL2Sum += normL2
			normL1Sum += normL1

		# average over all models
		normL2 = normL2Sum / self.numModels
		normL1 = normL1Sum / self.numModels

		# norm is sometimes zero, just default to -16 for perfect double-precision convergence I guess
		if (normL2 == 0.0):
			normOutL2 = -16.0
		else:
			normOutL2 = np.log10(normL2)
		
		if (normL1 == 0.0):
			normOutL1 = -16.0
		else: 
			normOutL1 = np.log10(normL1)

		outString = (str(subiter+1)+":\tL2: %18.14f, \tL1: %18.14f") % (normOutL2, normOutL1)
		print(outString)

		solDomain.solInt.resNormL2 = normL2
		solDomain.solInt.resNormL1 = normL1
		solDomain.solInt.resNormHistory[solver.iter-1, :] = [normL2, normL1]
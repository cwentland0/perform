
from pygems1d.inputFuncs import readInputFile, catchList, catchInput
from pygems1d.rom.linearProjROM.linearGalerkinProj import linearGalerkinProj
from pygems1d.rom.linearProjROM.linearSPLSVTProj import linearSPLSVTProj
from pygems1d.timeIntegrator.explicitIntegrator import rkExplicit
from pygems1d.timeIntegrator.implicitIntegrator import bdf
from pygems1d.solution.solutionPhys import solutionPhys
from pygems1d.spaceSchemes import calcRHS
from pygems1d.Jacobians import calcDResDSolPrim

import numpy as np
from time import sleep
import pdb
import os


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

		self.adaptiveROM = catchInput(romDict, "adaptiveROM", False)

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
		# TODO: move this selection to an external function? This is repeated in solutionDomain
		# TODO: timeScheme should be specific to the romDomain, not the solver
		if self.hasTimeIntegrator:
			if (solver.timeScheme == "bdf"):
				self.timeIntegrator = bdf(solver.paramDict)
			elif (solver.timeScheme == "rkExp"):
				self.timeIntegrator = rkExplicit(solver.paramDict)
			else:
				raise ValueError("Invalid choice of timeScheme: "+solver.timeScheme)

			# initialize code history
			# TODO: this is necessary for non-time-integrated methods, e.g. TCN
			for model in self.modelList:
				model.codeHist = [model.code.copy()] * (self.timeIntegrator.timeOrder+1)

		else:
			self.timeIntegrator = None 	# TODO: this might be pointless


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
		
			for self.timeIntegrator.subiter in range(1, self.timeIntegrator.subiterMax+1):

				self.advanceSubiter(solDomain, solver)
				
				if (self.timeIntegrator.timeType == "implicit"):
					solDomain.solInt.calcResNorms(solver, self.timeIntegrator.subiter)
					if (solDomain.solInt.resNormL2 < self.timeIntegrator.resTol): break

		solDomain.solInt.updateSolHist()
		self.updateCodeHist()


	def advanceSubiter(self, solDomain, solver):
		"""
		Advance physical solution forward one subiteration of time integrator
		"""

		solInt = solDomain.solInt
		res, resJacob = None, None

		if self.isIntrusive:
			calcRHS(solDomain, solver)

		if (self.timeIntegrator.timeType == "implicit"):

			raise ValueError("Implicit ROM under development")

			if self.isIntrusive:
				res = self.timeIntegrator.calcResidual(solInt.solHistCons, solInt.RHS, solver)
				resJacob = calcDResDSolPrim(solDomain, solver)

			for modelIdx, model in enumerate(self.modelList):
				dCode = model.calcDCode(resJacob, res)
				model.code += dCode
				model.codeHist[0] = model.code.copy()
				model.updateSol(solDomain)
				
			dSol = solInt.solPrim - solInt.solHistPrim[0]
			res = resJacob @ dSol.ravel("F") - solInt.res.ravel("F")
			solInt.res = np.reshape(res, (solDomain.gasModel.numEqs, solver.mesh.numCells), order='F')

			solInt.updateState(fromCons=False) 	

		else:

			for modelIdx, model in enumerate(self.modelList):

				model.calcRHSLowDim(self, solDomain)
				dCode = self.timeIntegrator.solveSolChange(model.rhsLowDim)
				model.code = model.codeHist[0] + dCode
				model.updateSol(solDomain)
			
			solInt.updateState(fromCons=True)

	def updateCodeHist(self):
		"""
		Update low-dimensional state history after physical time step
		"""

		for model in self.modelList:

			model.codeHist[1:] = model.codeHist[:-1]
			model.codeHist[0]  = model.code.copy()

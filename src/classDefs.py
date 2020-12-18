import numpy as np 
import constants
from constants import realType, RUniv
from inputFuncs import readInputFile
from math import floor, log
import os
import pdb

# inputs/solver properties
class parameters:

	def __init__(self, workdir, paramFile):
		self.workdir = workdir

		paramDict = readInputFile(paramFile)
		self.paramDict = paramDict

		# input files, output directories
		self.gasFile 		= str(paramDict["gasFile"]) 		# gas properties file (string)
		self.meshFile 		= str(paramDict["meshFile"]) 		# mesh properties file (string)
		try:
			self.initFile	= str(paramDict["initFile"])		# initial condition file
		except:
			self.initFile 	= None

		self.unsOutDir 		= os.path.join(workdir, constants.unsteadyOutputDir)
		self.probeOutDir 	= os.path.join(workdir, constants.probeOutputDir)
		self.imgOutDir 		= os.path.join(workdir, constants.imageOutputDir)
		self.restOutDir 	= os.path.join(workdir, constants.restartOutputDir)

		# temporal discretization parameters
		self.runSteady 		= catchInput(paramDict, "runSteady", False) # whether to run "steady" simulation, just reporting solution change norm
		self.dt 			= float(paramDict["dt"])					# physical time step
		self.dtau 			= catchInput(paramDict, "dtau", 1.e-5)		# psuedo time step
		self.numSteps 		= int(paramDict["numSteps"])	# total number of physical time iterations
		self.timeScheme 	= str(paramDict["timeScheme"]) 	# time integration scheme (string)
		self.timeOrder 		= int(paramDict["timeOrder"])	# time integration order of accuracy (int)
		self.solTime 		= 0.0

		# robustness controls
		self.adaptDTau 		= catchInput(paramDict, "adaptDTau", False)	# whether to compute adaptive pseudo time step
		self.CFL 			= catchInput(paramDict, "CFL", 10) 			# reference CFL for advective control of dtau
		self.VNN 			= catchInput(paramDict, "VNN", 20) 			# von Neumann number for diffusion control of dtau
		self.refConst 		= catchInput(paramDict, "refConst", [None])  	# constants for limiting dtau	
		self.relaxConst 	= catchInput(paramDict, "relaxConst", [None]) 	#

		if (self.timeScheme in ["bdf","pTime"]):
			self.timeType 		= "implicit"
			if self.runSteady: self.steadyThresh = catchInput(paramDict, "steadyThresh", -10.0) # exponent threshold on steady residual
			self.numSubIters 	= catchInput(paramDict, "numSubIters", 50)	# maximum number of subiterations for iterative solver
			self.resTol 		= catchInput(paramDict, "resTol", 1e-12)	# residual tolerance for iterative solver 
		elif (self.timeScheme == "rk"):
			if (self.runSteady): raise ValueError("Cannot run steady-state solution with explicit time integrator!")
			self.timeType 		= "explicit"
			self.numSubIters 	= self.timeOrder
			self.subIterCoeffs 	= constants.rkCoeffs[-self.timeOrder:]

		# spatial discretization parameters
		self.spaceScheme 	= catchInput(paramDict, "spaceScheme", "roe")	# spatial discretization scheme (string)
		self.spaceOrder 	= catchInput(paramDict, "spaceOrder", 1)		# spatial discretization order of accuracy (int)
		self.viscScheme 	= catchInput(paramDict, "viscScheme", 0)		# 0 for inviscid, 1 for viscous

		# misc
		self.velAdd 		= catchInput(paramDict, "velAdd", 0.0)
		self.steadyNormPrim	= catchInput(paramDict, "steadyNorm", [None])
		self.sourceOn 		= catchInput(paramDict, "sourceOn", True)

		# Solving for Primitive Variables
		self.solforPrim 		= catchInput(paramDict, "solforPrim", False)

		# restart files
		self.saveRestarts 	= catchInput(paramDict, "saveRestarts", False) 	# whether to save restart files
		if self.saveRestarts:
			self.restartInterval 	= catchInput(paramDict, "restartInterval", 100)	# number of steps between restart file saves
			self.numRestarts 		= catchInput(paramDict, "numRestarts", 20) 		# number of restart files to keep saved
			self.restartIter 		= 1		# file number counter
		self.initFromRestart = catchInput(paramDict, "initFromRestart", False)

		if ((self.initFile == None) and (not self.initFromRestart)):
			try:
				self.icParamsFile 	= str(paramDict["icParamsFile"])
			except:
				raise KeyError("If not providing IC profile or restarting from restart file, must provide icParamsFile")

		# unsteady output
		self.outInterval	= catchInput(paramDict, "outInterval", 1) 		# iteration interval to save data (int)
		self.primOut		= catchInput(paramDict, "primOut", True)		# whether to save the primitive variables
		self.consOut 		= catchInput(paramDict, "consOut", False) 		# whether to save the conservative variables
		self.sourceOut 		= catchInput(paramDict, "sourceOut", False) 	# whether to save the species source term
		self.RHSOut 		= catchInput(paramDict, "RHSOut", False)		# whether to save the RHS vector
		self.numSnaps 		= int(self.numSteps / self.outInterval)

		# visualization
		self.visType 		= catchInput(paramDict, "visType", "None")			# "field" or "point"
		if (self.visType != "None"): 
			self.visVar		= catchList(paramDict, "visVar", [None])		# variable(s) to visualize (string)
			self.numVis 	= len(self.visVar)
			if (None in self.visVar):
				raise KeyError("If requesting visualization, must provide list of valid visualization variables")
			
			else:
				self.visXBounds 	= catchList(paramDict, "visXBounds", [[None,None]], lenHighest=self.numVis)  # x-axis plot bounds
				self.visYBounds 	= catchList(paramDict, "visYBounds", [[None,None]], lenHighest=self.numVis)  # y-axis plot bounds
				self.visInterval 	= catchInput(paramDict, "visInterval", 1)			# interval at which to visualize (int)
				self.visSave 		= catchInput(paramDict, "visSave", False)			# whether to write images to disk (bool)

		# TODO: account for 0, 2+ probes
		self.probeLoc 		= float(paramDict["probeLoc"])						# point monitor location (will reference closest cell)
		self.probeSec 		= "interior"										# "interior", "inlet", or "outlet", changes later depending on probeLoc
		numImgs 			= int(self.numSteps / self.visInterval)
		self.imgString 		= '%0'+str(floor(log(numImgs, 10))+1)+'d'	# TODO: this fails if numSteps is less than visInterval

		# ROM parameters
		# TODO: potentially move this to another input file
		self.calcROM 		= catchInput(paramDict, "calcROM", False)
		if not self.calcROM: 
			self.simType = "FOM"
		else:
			self.romInputs = os.path.join(workdir, constants.romInputs)

# gas thermofluid properties
# TODO: expand Arrhenius factors to allow for multiple reactions
class gasProps:

	def __init__(self, gasFile):
		gasDict = readInputFile(gasFile)

		# gas composition
		self.numSpeciesFull 	= int(gasDict["numSpecies"])				# total number of species in case
		self.molWeights 		= gasDict["molWeights"].astype(realType)	# molecular weights, g/mol
		self.enthRef 			= gasDict["enthRef"].astype(realType) 		# reference enthalpy, J/kg
		self.tempRef 			= gasDict["tempRef"]						# reference temperature, K
		self.Cp 				= gasDict["Cp"].astype(realType)			# heat capacity at constant pressure, J/(kg-K)
		self.Pr 				= gasDict["Pr"].astype(realType)			# Prandtl number
		self.Sc 				= gasDict["Sc"].astype(realType)			# Schmidt number
		self.muRef				= gasDict["muRef"].astype(realType)		# reference viscosity for Sutherland model (I think?)
		
		# Arrhenius factors
		# TODO: modify these to allow for multiple global reactions
		self.nu 				= gasDict["nu"].astype(realType)		# ?????
		self.nuArr 				= gasDict["nuArr"].astype(realType)	# ?????
		self.actEnergy			= float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		self.preExpFact 		= float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		self.RGas 				= RUniv / self.molWeights 			# specific gas constant, J/(K*kg)
		self.numSpecies 		= self.numSpeciesFull - 1			# last species is not directly solved for
		self.numEqs 			= self.numSpecies + 3				# pressure, velocity, temperature, and species transport
		self.molWeightNu 		= self.molWeights * self.nu 
		self.mwInvDiffs 		= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]


# mesh properties
# TODO: could expand to non-uniform meshes
class geometry:

	def __init__(self, meshFile):
		meshDict = readInputFile(meshFile)

		# domain definition
		self.xL 				= float(meshDict["xL"])
		self.xR 				= float(meshDict["xR"])
		self.numCells 	= int(meshDict["numCells"])

		# mesh coordinates
		self.xFace 	= np.linspace(self.xL, self.xR, self.numCells + 1, dtype = realType)
		self.xCell 	= (self.xFace[1:] + self.xFace[:-1]) / 2.0
		self.dx 		= self.xFace[1] - self.xFace[0]
		self.numNodes 	= self.numCells + 1

# assign default values if user does not provide a certain input
# TODO: correct error handling if default type is not recognized
# TODO: instead of trusting user for NoneType, could also use NaN/Inf to indicate int/float defaults without passing a numerical default
# 		or could just pass the actual default type lol, that'd be easier
def catchInput(inDict, inKey, default):

	defaultType = type(default)
	try:
		# if NoneType passed as default, trust user
		if (defaultType == type(None)):
			val = inDict[inKey]
		else:
			val = defaultType(inDict[inKey])

	# if inDict doesn't contain the given key, fall back to given default 
	except:
		val = default

	return val

# input processor for reading lists or lists of lists
# default defines length of lists at lowest level
# TODO: could make a recursive function probably, just hard to define appropriate list lengths at each level
def catchList(inDict, inKey, default, lenHighest=1):

	listOfListsFlag = (type(default[0]) == list)
	
	try:
		inList = inDict[inKey]

		# list of lists
		if listOfListsFlag:
			typeDefault = type(default[0][0])
			valList = []
			for listIdx in range(lenHighest):
				# if default type is NoneType, trust user
				if (typeDefault == type(None)):
					valList.append(inList[listIdx])
				else:
					castInList = [typeDefault(inVal) for inVal in inList[listIdx]]
					valList.append(castInList)

		# normal list
		else:
			typeDefault = type(default[0])
			# if default type is NoneType, trust user 
			if (typeDefault == type(None)):
				valList = inList
			else:
				valList = [typeDefault(inVal) for inVal in inList]

	except:
		if listOfListsFlag:
			valList = []
			for listIdx in range(lenHighest):
				valList.append(default[0])
		else:
			valList = default

	return valList
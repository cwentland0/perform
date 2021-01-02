import constants
from inputFuncs import readInputFile, catchInput, catchList
import timeIntegrator
import gasModel
import mesh
import numpy as np 
from math import floor, log
import os
import pdb


class systemSolver:
	"""
	Container class for input parameters, domain geometry, spatial/temporal discretization, and gas model
	"""

	def __init__(self):

		# input parameters from solverParams.inp
		paramFile = os.path.join(constants.workingDir, constants.paramInputs)
		paramDict = readInputFile(paramFile)
		self.paramDict = paramDict

		# gas model
		gasFile = str(paramDict["gasFile"]) 		# gas properties file (string)
		gasDict = readInputFile(gasFile) 
		gasType = catchInput(gasDict, "gasType", "cpg")
		if (gasType == "cpg"):
			self.gasModel = gasModel.caloricallyPerfectGas(gasDict)
		else:
			raise ValueError("Ivalid choice of gasType: " + gasType)

		# spatial domain
		meshFile 	= str(paramDict["meshFile"]) 		# mesh properties file (string)
		meshDict 	= readInputFile(meshFile)
		self.mesh 	= mesh.mesh(meshDict)
		# TODO: selection for different meshes, when implemented
		# meshType 	= str(meshDict["meshType"])
		# if (meshType == "uniform"):
		# 	self.mesh 	= mesh.uniformMesh(meshDict)
		# else:
		# 	raise ValueError("Invalid choice of meshType: " + meshType)

		# initial condition file
		try:
			self.initFile	= str(paramDict["initFile"]) 
		except:
			self.initFile 	= None

		# temporal discretization
		timeScheme 			= str(paramDict["timeScheme"])
		self.solTime 		= 0.0

		if (timeScheme == "bdf"):
			self.timeIntegrator = timeIntegrator.bdf(paramDict)
		elif (timeScheme == "rkExp"):
			self.timeIntegrator = timeIntegrator.rkExplicit(paramDict)
		else:
			raise ValueError("Invalid choice of timeScheme: "+timeScheme)

		if self.timeIntegrator.runSteady:
			if (self.timeIntegrator.timeType == "implicit"):
				self.steadyThresh = catchInput(paramDict, "steadyThresh", -10.0) # exponent threshold on steady residual
			else:
				raise ValueError("Cannot run steady-state solution with explicit time integrator!")
		

		# spatial discretization parameters
		self.spaceScheme 	= catchInput(paramDict, "spaceScheme", "roe")	# spatial discretization scheme (string)
		self.spaceOrder 	= catchInput(paramDict, "spaceOrder", 1)		# spatial discretization order of accuracy (int)
		self.gradLimiter 	= catchInput(paramDict, "gradLimiter", 0)		# gradient limiter for higher-order face reconstructions
		self.viscScheme 	= catchInput(paramDict, "viscScheme", 0)		# 0 for inviscid, 1 for viscous

		# misc
		self.velAdd 		= catchInput(paramDict, "velAdd", 0.0)
		self.resNormPrim	= catchInput(paramDict, "steadyNorm", [None])
		self.sourceOn 		= catchInput(paramDict, "sourceOn", True)

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
		self.numSnaps 		= int(self.timeIntegrator.numSteps / self.outInterval)

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
		numImgs 			= int(self.timeIntegrator.numSteps / self.visInterval)
		self.imgString 		= '%0'+str(floor(log(numImgs, 10))+1)+'d'	# TODO: this fails if numSteps is less than visInterval

		# ROM flag
		self.calcROM 		= catchInput(paramDict, "calcROM", False)
		if not self.calcROM: 
			self.simType = "FOM"
		else:
			self.romInputs = os.path.join(workdir, constants.romInputs)
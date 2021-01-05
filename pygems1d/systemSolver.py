import pygems1d.constants as const
from pygems1d.inputFuncs import readInputFile, catchInput, catchList
from pygems1d.timeIntegrator.explicitIntegrator import rkExplicit
from pygems1d.timeIntegrator.implicitIntegrator import bdf
from pygems1d.gasModel.caloricallyPerfectGas import caloricallyPerfectGas
import pygems1d.mesh as mesh

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
		paramFile = os.path.join(const.workingDir, const.paramInputs)
		paramDict = readInputFile(paramFile)
		self.paramDict = paramDict

		# gas model
		gasFile = str(paramDict["gasFile"]) 		# gas properties file (string)
		gasDict = readInputFile(gasFile) 
		gasType = catchInput(gasDict, "gasType", "cpg")
		if (gasType == "cpg"):
			self.gasModel = caloricallyPerfectGas(gasDict)
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
			self.timeIntegrator = bdf(paramDict)
		elif (timeScheme == "rkExp"):
			self.timeIntegrator = rkExplicit(paramDict)
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

		# restart files
		# TODO: could move this to solutionDomain, not terribly necessary
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

		# misc
		self.velAdd 		= catchInput(paramDict, "velAdd", 0.0)
		self.resNormPrim	= catchInput(paramDict, "steadyNorm", [None])
		self.sourceOn 		= catchInput(paramDict, "sourceOn", True)
		self.solveFailed 	= False

		# ROM flag
		self.calcROM = catchInput(paramDict, "calcROM", False)
		if not self.calcROM: 
			self.simType = "FOM"
		else:
			self.simType = "ROM"
			self.romInputs = os.path.join(workdir, const.romInputs)
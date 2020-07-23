import numpy as np 
from constants import rkCoeffs, floatType
from inputFuncs import readInputFile
import pdb

# TODO: honestly, some of this could just be permanently dicts if we really wanted
#	guess it's more readable with object references though
# TODO: error checks for input values
# TODO: add type descriptors for every parameter
# TODO: cast input parameters appropriately (mostly all ints)

# inputs/solver properties
class parameters:

	def __init__(self, paramFile):
		paramDict = readInputFile(paramFile)
		self.paramDict = paramDict

		# input files
		self.gasFile 		= str(paramDict["gasFile"]) 		# gas properties file (string)
		self.meshFile 		= str(paramDict["meshFile"]) 	# mesh properties file (string)
		self.initFile		= str(paramDict["initFile"])		# initial condition file

		# temporal discretization parameters
		self.dt 			= float(paramDict["dt"])		# physical time step
		self.numSteps 		= int(paramDict["numSteps"])	# total number of physical time iterations
		self.timeScheme 	= str(paramDict["timeScheme"]) 	# time integration scheme (string)
		self.timeOrder 		= int(paramDict["timeOrder"])	# time integration order of accuracy (int)
		
		if (self.timeScheme in ["bdf","pTime"]):
			self.timeType 		= "implicit"
			self.numSubIters 	= int(paramDict["numSubIters"])	# maximum number of subiterations for iterative solver
			self.resTol 		= float(paramDict["resTol"])	# residual tolerance for iterative solver 
		elif (self.timeScheme == "rk"):
			self.timeType 		= "explicit"
			self.numSubIters 	= self.timeOrder
			self.subIterCoeffs 	= rkCoeffs[-self.timeOrder:]

		# spatial discretization parameters
		self.spaceScheme 	= str(paramDict["spaceScheme"])	# spatial discretization scheme (string)
		self.spaceOrder 	= int(paramDict["spaceOrder"])	# spatial discretization order of accuracy (int)
		self.viscScheme 	= int(paramDict["viscScheme"])	# 0 for inviscid, 1 for viscous
		
		# misc
		self.velAdd 		= float(paramDict["velAdd"]) 	# velocity to add to entire initial condition

		# output
		self.outInterval	= int(paramDict["outInterval"]) 			# iteration interval to save data (int)
		self.primOut		= bool(paramDict["primOut"])				# whether to save the primitive variables
		self.consOut 		= bool(paramDict["consOut"]) 				# whether to save the conservative variables
		self.RHSOut 		= bool(paramDict["RHSOut"])					# whether to save the RHS vector
		self.numSnaps 		= int(self.numSteps / self.outInterval)

		self.visType 		= str(paramDict["visType"])				# "field" or "point"
		self.visVar			= str(paramDict["visVar"])				# variable to visualize (string)
		self.visInterval 	= int(paramDict["visInterval"])			# interval at which to visualize (int)
		if (self.visType == "probe"):
			self.probeLoc = float(paramDict["probeLoc"])			# point monitor location (will reference closest cell)

		self.solTime = 0.0		


# gas thermofluid properties
# TODO: expand Arrhenius factors to allow for multiple reactions
class gasProps:

	def __init__(self, gasFile):
		gasDict = readInputFile(gasFile)

		# gas composition
		self.numSpecies_full 	= int(gasDict["numSpecies"])				# total number of species in case
		self.molWeights 		= gasDict["molWeights"].astype(floatType)	# molecular weights, g/mol
		self.enthRef 			= gasDict["enthRef"].astype(floatType) 		# reference enthalpy, J/kg
		self.tempRef 			= gasDict["tempRef"]						# reference temperature, K
		self.Cp 				= gasDict["Cp"].astype(floatType)			# heat capacity at constant pressure, J/K
		self.Pr 				= gasDict["Pr"].astype(floatType)			# Prandtl number
		self.Sc 				= gasDict["Sc"].astype(floatType)			# Schmidt number
		self.muRef				= gasDict["muRef"].astype(floatType)		# reference viscosity for Sutherland model (I think?)
		
		# Arrhenius factors
		# TODO: modify these to allow for multiple global reactions
		self.nu 				= gasDict["nu"].astype(floatType)		# ?????
		self.nuArr 				= gasDict["nuArr"].astype(floatType)	# ?????
		self.actEnergy			= float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		self.preExpFact 		= float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		self.RGas 				= 1.0 / self.molWeights 			# specific gas constant, J/(K*mol) * 1,000
		self.numSpecies 		= self.numSpecies_full - 1			# last species is not directly solved for
		self.numEqs 			= self.numSpecies + 3				# pressure, velocity, temperature, and species transport
		self.molWeightNu 		= self.molWeights * self.nu 
		self.mwDiffs 			= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]


# mesh properties
# TODO: could expand to non-uniform meshes
class geometry:

	def __init__(self, meshFile):
		meshDict = readInputFile(meshFile)

		# domain definition
		xL 				= float(meshDict["xL"])
		xR 				= float(meshDict["xR"])
		self.numCells 	= int(meshDict["numCells"])

		# mesh coordinates
		self.x_node 	= np.linspace(xL, xR, self.numCells + 1, dtype = floatType)
		self.x_cell 	= (self.x_node[1:] + self.x_node[:-1]) / 2.0
		self.dx 		= self.x_node[1] - self.x_node[0]
		self.numNodes 	= self.numCells + 1




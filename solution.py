import numpy as np
from math import sin, pi, floor
import constants
from classDefs import parameters, gasProps, geometry
from inputFuncs import parseBC, readInputFile
import stateFuncs
import pdb

# TODO error checking on initial solution load

# full-domain solution
class solutionPhys:

	# TODO: when you move the boundary class out of params, can reference numSnaps and primOut, etc. from that
	#		Currently can't import parameters class due to circular dependency
	def __init__(self, numCells, solPrimIn, solConsIn, gas: gasProps, params: parameters):
		
		# solution and mixture properties
		self.solPrim	= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# solution in primitive variables
		self.solCons	= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# solution in conservative variables
		self.RHS 		= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# RHS function
		self.mwMix 		= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# mixture molecular weight
		self.RMix		= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# mixture specific gas constant
		self.gammaMix 	= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# mixture ratio of specific heats
		self.enthRefMix = np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# mixture reference enthalpy
		self.CpMix 		= np.zeros((numCells, gas.numEqs), dtype = constants.floatType)		# mixture specific heat at constant pressure

		# load initial condition and check size
		assert(solPrimIn.shape == (numCells, gas.numEqs))
		assert(solConsIn.shape == (numCells, gas.numEqs))
		self.solPrim = solPrimIn
		self.solCons = solConsIn

		# snapshot storage matrices, store initial condition
		if params.primOut: 
			self.primSnap = np.zeros((numCells, gas.numEqs, params.numSnaps+1), dtype = constants.floatType)
			self.primSnap[:,:,0] = solPrimIn 
		if params.consOut: 
			self.consSnap = np.zeros((numCells, gas.numEqs, params.numSnaps+1), dtype = constants.floatType)
			self.consSnap[:,:,0] = solConsIn
		if params.RHSOut:  self.RHSSnap  = np.zeros((numCells, gas.numEqs, params.numSnaps), dtype = constants.floatType)

		# TODO: initialize thermo properties at initialization too

	# update solution after a time step 
	# TODO: convert to using class methods
	def updateState(self, gas, fromCons: bool = True):

		if fromCons:
			self.solPrim, self.RMix, self.enthRefMix, self.CpMix = stateFuncs.calcStateFromCons(self.solCons, gas)
		else:
			self.solCons, self.RMix, self.enthRefMix, self.CpMix = stateFuncs.calcStateFromPrim(self.solPrim, gas)

# grouping class for boundaries
class boundaries:
	
	def __init__(self, sol: solutionPhys, params: parameters, gas: gasProps):
		# initialize inlet
		pressIn, velIn, tempIn, massFracIn, pertTypeIn, pertPercIn, pertFreqIn = parseBC("inlet", params.paramDict)
		self.inlet = boundary(params.paramDict["boundType_inlet"], pressIn, velIn, tempIn, massFracIn, 
								pertTypeIn, pertPercIn, pertFreqIn)
		self.inlet.initState(sol.solPrim[[0],:], sol.solCons[[0],:], gas, params)
		self.inlet.sol.updateState(gas, fromCons = False)

		# initialize outlet
		pressOut, velOut, tempOut, massFracOut, pertTypeOut, pertPercOut, pertFreqOut = parseBC("outlet", params.paramDict)
		self.outlet = boundary(params.paramDict["boundType_outlet"], pressOut, velOut, tempOut, massFracOut, 
								pertTypeOut, pertPercOut, pertFreqOut)
		self.outlet.initState(sol.solPrim[[-1],:], sol.solCons[[-1],:], gas, params)
		self.outlet.sol.updateState(gas, fromCons = False)

# boundary cell parameters
class boundary:

	def __init__(self, boundType, press, vel, temp, massFrac, pertType, pertPerc, pertFreq):
		
		self.type 		= boundType 
		self.press 		= press
		self.vel 		= vel 
		self.temp 		= temp 
		self.massFrac 	= massFrac

		self.pertType 	= pertType 
		self.pertPerc 	= pertPerc 
		self.pertFreq 	= pertFreq

	# TODO: need to init state to something besides zero if specifying full state for BC
	def initState(self, solPrim, solCons, gas: gasProps, params: parameters):
		# solPrimDummy = np.zeros((1, gas.numEqs), dtype = constants.floatType)
		# solConsDummy = np.zeros((1, gas.numEqs), dtype = constants.floatType)
		self.rho 	= solCons[0,0]
		self.vel 	= solPrim[0,1]
		self.temp 	= solPrim[0,2]
		self.sol 	= solutionPhys(1, solPrim, solCons, gas, params) # not saving any snapshots here

	# compute sinusoidal perturbation
	# TODO: add phase offset
	def calcPert(self, t):
		
		pert = 0.0
		for f in self.pertFreq:
			pert += sin(2.0 * pi * self.pertFreq * t)
		pert *= self.pertPerc 

		return pert

# low-dimensional ROM solution
# there is one instance of solution_rom attached to each decoder
# class solution_rom:

# generate "left" and "right" states
# TODO: check for existence of initialConditionParams.inp
def genInitialCondition(params: parameters, gas: gasProps, geom: geometry):

	icDict 	= readInputFile(params.icParamsFile)

	splitIdx 	= np.absolute(geom.x_cell - icDict["xSplit"]).argmin()
	solPrim 	= np.zeros((geom.numCells, gas.numEqs), dtype = constants.floatType)

	# left state
	solPrim[:splitIdx,0] 	= icDict["pressLeft"]
	solPrim[:splitIdx,1] 	= icDict["velLeft"]
	solPrim[:splitIdx,2] 	= icDict["tempLeft"]
	solPrim[:splitIdx,3:] 	= icDict["massFracLeft"][:-1]

	# right state
	solPrim[splitIdx:,0] 	= icDict["pressRight"]
	solPrim[splitIdx:,1] 	= icDict["velRight"]
	solPrim[splitIdx:,2] 	= icDict["tempRight"]
	solPrim[splitIdx:,3:] 	= icDict["massFracRight"][:-1]
	
	solCons, _, _, _ = stateFuncs.calcStateFromPrim(solPrim, gas)

	# initCond = np.concatenate((solPrim[:,:,np.newaxis], solCons[:,:,np.newaxis]), axis = 2)

	return solPrim, solCons
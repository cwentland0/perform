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
		self.solPrim	= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# solution in primitive variables
		self.solCons	= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# solution in conservative variables
		self.RHS 		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# RHS function
		self.mwMix 		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture molecular weight
		self.RMix		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture specific gas constant
		self.gammaMix 	= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture ratio of specific heats
		self.enthRefMix = np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture reference enthalpy
		self.CpMix 		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture specific heat at constant pressure

		# load initial condition and check size
		assert(solPrimIn.shape == (numCells, gas.numEqs))
		assert(solConsIn.shape == (numCells, gas.numEqs))
		self.solPrim = solPrimIn.copy()
		self.solCons = solConsIn.copy()

		# snapshot storage matrices, store initial condition
		if params.primOut: 
			self.primSnap = np.zeros((numCells, gas.numEqs, params.numSnaps+1), dtype = constants.realType)
			self.primSnap[:,:,0] = solPrimIn 
		if params.consOut: 
			self.consSnap = np.zeros((numCells, gas.numEqs, params.numSnaps+1), dtype = constants.realType)
			self.consSnap[:,:,0] = solConsIn
		if params.RHSOut:  self.RHSSnap  = np.zeros((numCells, gas.numEqs, params.numSnaps), dtype = constants.realType)

		# TODO: initialize thermo properties at initialization too (might be problematic for BC state)

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
		pressIn, velIn, tempIn, massFracIn, rhoIn, pertTypeIn, pertPercIn, pertFreqIn = parseBC("inlet", params.paramDict)
		self.inlet = boundary(params.paramDict["boundType_inlet"], params, gas, pressIn, velIn, tempIn, massFracIn, rhoIn,
								pertTypeIn, pertPercIn, pertFreqIn)

		# initialize outlet
		pressOut, velOut, tempOut, massFracOut, rhoOut, pertTypeOut, pertPercOut, pertFreqOut = parseBC("outlet", params.paramDict)
		self.outlet = boundary(params.paramDict["boundType_outlet"], params, gas, pressOut, velOut, tempOut, massFracOut, rhoOut,
								pertTypeOut, pertPercOut, pertFreqOut)


# boundary cell parameters
class boundary:

	def __init__(self, boundType, params, gas: gasProps, press, vel, temp, massFrac, rho, pertType, pertPerc, pertFreq):
		
		self.type 		= boundType 

		# this generally stores fixed/stagnation properties
		self.press 		= press
		self.vel 		= vel 
		self.temp 		= temp 
		self.massFrac 	= massFrac 
		self.rho 		= rho
		self.CpMix 		= stateFuncs.calcCpMixture(self.massFrac[:-1], gas)
		self.RMix 		= stateFuncs.calcGasConstantMixture(self.massFrac[:-1], gas)
		self.gamma 		= stateFuncs.calcGammaMixture(self.RMix, self.CpMix)
		self.enthRefMix = stateFuncs.calcEnthRefMixture(self.massFrac[:-1], gas)

		# boundary flux for weak BCs
		if params.weakBCs:
			self.flux 		= np.zeros(gas.numEqs, dtype = constants.realType)
		
		# unsteady ghost cell state for strong BCs
		else:
			solDummy 		= np.zeros((1, gas.numEqs), dtype = constants.realType)
			self.sol 		= solutionPhys(1, solDummy, solDummy, gas, params)
			self.sol.solPrim[0,3:] = self.massFrac[:-1]

		self.pertType 	= pertType 
		self.pertPerc 	= pertPerc 
		self.pertFreq 	= pertFreq

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
	solPrim 	= np.zeros((geom.numCells, gas.numEqs), dtype = constants.realType)

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
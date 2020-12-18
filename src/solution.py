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
		self.source 	= np.zeros((numCells, gas.numSpecies), dtype = constants.realType)	# reaction source term
		self.RHS 		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# RHS function
		self.mwMix 		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture molecular weight
		self.RMix		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture specific gas constant
		self.gammaMix 	= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture ratio of specific heats
		self.enthRefMix = np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture reference enthalpy
		self.CpMix 		= np.zeros((numCells, gas.numEqs), dtype = constants.realType)		# mixture specific heat at constant pressure

		if (params.adaptDTau):
			self.srf 	= np.zeros(numCells, dtype = constants.realType)

		# residual, time history, residual normalization for implicit methods
		if (params.timeType == "implicit"):
			self.res 			= np.zeros((numCells, gas.numEqs), dtype = constants.realType)
			self.solHistCons 	= [np.zeros((numCells, gas.numEqs), dtype = constants.realType)]*(params.timeOrder+1)
			self.solHistPrim 	= [np.zeros((numCells, gas.numEqs), dtype = constants.realType)]*(params.timeOrder+1)

			# normalizations for "steady" solution residual norms
			if (params.runSteady):
				self.resOutL2 = 0.0
				self.resOutL1 = 0.0

				# if nothing was provided
				if ((len(params.steadyNormPrim) == 1) and (params.steadyNormPrim[0] == None)):
					params.steadyNormPrim = [None]*gas.numEqs
				# check user input
				else:
					assert(len(params.steadyNormPrim) == gas.numEqs)

				# replace any None's with defaults
				for varIdx in range(gas.numEqs):
					if (params.steadyNormPrim[varIdx] == None):
						# 0: pressure, 1: velocity, 2: temperature, >=3: species
						params.steadyNormPrim[varIdx] = constants.steadyNormPrimDefault[min(varIdx,3)]

		# load initial condition and check size
		assert(solPrimIn.shape == (numCells, gas.numEqs))
		assert(solConsIn.shape == (numCells, gas.numEqs))
		self.solPrim = solPrimIn.copy()
		self.solCons = solConsIn.copy()

		# snapshot storage matrices, store initial condition
		if params.primOut: 
			self.primSnap = np.zeros((numCells, gas.numEqs, params.numSnaps+1), dtype = constants.realType)
			self.primSnap[:,:,0] = solPrimIn.copy()
		if params.consOut: 
			self.consSnap = np.zeros((numCells, gas.numEqs, params.numSnaps+1), dtype = constants.realType)
			self.consSnap[:,:,0] = solConsIn.copy()
		if params.sourceOut: self.sourceSnap = np.zeros((numCells, gas.numSpecies, params.numSnaps+1), dtype = constants.realType)
		if params.RHSOut:  self.RHSSnap  = np.zeros((numCells, gas.numEqs, params.numSnaps), dtype = constants.realType)

		# TODO: initialize thermo properties at initialization too (might be problematic for BC state)

	# initialize time history of solution
	def initSolHist(self, params: parameters):
		
		for i in range(params.timeOrder+1):
			self.solHistCons[i] = self.solCons.copy()
			self.solHistPrim[i] = self.solPrim.copy()
	
	# update time history of solution for implicit time integration
	def updateSolHist(self):
		self.solHistCons[1:] = self.solHistCons[:-1]
		self.solHistPrim[1:] = self.solHistPrim[:-1]
			
	# update solution after a time step 
	# TODO: convert to using class methods
	def updateState(self, gas, fromCons: bool = True):

		if fromCons:
			self.solPrim, self.RMix, self.enthRefMix, self.CpMix = stateFuncs.calcStateFromCons(self.solCons, gas)
		else:
			self.solCons, self.RMix, self.enthRefMix, self.CpMix = stateFuncs.calcStateFromPrim(self.solPrim, gas)

	# print residual norms
	# TODO: do some decent formatting on output, depending on resType
	def resOutput(self, params: parameters, tStep):

		dSol = self.solHistPrim[1] - self.solHistPrim[2]
		dSolAbs = np.abs(dSol)

		# L2 norm
		resSumL2 = np.sum(np.square(dSolAbs), axis=0) 		# sum of squares
		resSumL2[:] /= dSol.shape[0]   							# divide by number of cells
		resSumL2 /= np.square(params.steadyNormPrim) 			# divide by square of normalization constants
		resSumL2 = np.sqrt(resSumL2) 		 					# square root
		resLogL2 = np.log10(resSumL2) 							# get exponent
		resOutL2 = np.mean(resLogL2)
			
		# L1 norm
		resSumL1 = np.sum(dSolAbs, axis=0) 				# sum of absolute values
		resSumL1[:] /= dSol.shape[0]   							# divide by number of cells
		resSumL1 /= params.steadyNormPrim 						# divide by normalization constants
		resLogL1 = np.log10(resSumL1) 							# get exponent
		resOutL1 = np.mean(resLogL1)

		print(str(tStep+1)+":\tL2 norm: "+str(resOutL2)+",\tL1 norm:"+str(resOutL1))

		self.resOutL2 = resOutL2 
		self.resOutL1 = resOutL1

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

		# ghost cell 
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

	splitIdx 	= np.absolute(geom.xCell - icDict["xSplit"]).argmin()
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
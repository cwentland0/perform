import numpy as np
from math import sin, pi, floor
import constants
from gasModel import gasModel
from inputFuncs import parseBC, readInputFile
import stateFuncs
import os
import pdb

# TODO error checking on initial solution load

# full-domain solution
class solutionPhys:

	def __init__(self, solPrimIn, solConsIn, numCells, solver):
		
		gas 		= solver.gasModel
		timeInt 	= solver.timeIntegrator

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


		# residual, time history, residual normalization for implicit methods
		if (timeInt.timeType == "implicit"):

			if ((timeInt.dualTime) and (solver.adaptDTau)):
				self.srf 	= np.zeros(numCells, dtype = constants.realType)

			self.res 			= np.zeros((numCells, gas.numEqs), dtype = constants.realType)
			self.solHistCons 	= [np.zeros((numCells, gas.numEqs), dtype = constants.realType)]*(timeInt.timeOrder+1)
			self.solHistPrim 	= [np.zeros((numCells, gas.numEqs), dtype = constants.realType)]*(timeInt.timeOrder+1)

			# normalizations for "steady" solution residual norms
			if (solver.runSteady):
				self.resOutL2 = 0.0
				self.resOutL1 = 0.0

				# if nothing was provided
				if ((len(solver.steadyNormPrim) == 1) and (solver.steadyNormPrim[0] == None)):
					solver.steadyNormPrim = [None]*gas.numEqs
				# check user input
				else:
					assert(len(solver.steadyNormPrim) == gas.numEqs)

				# replace any None's with defaults
				for varIdx in range(gas.numEqs):
					if (solver.steadyNormPrim[varIdx] == None):
						# 0: pressure, 1: velocity, 2: temperature, >=3: species
						solver.steadyNormPrim[varIdx] = constants.steadyNormPrimDefault[min(varIdx,3)]

		# load initial condition and check size
		assert(solPrimIn.shape == (numCells, gas.numEqs))
		assert(solConsIn.shape == (numCells, gas.numEqs))
		self.solPrim = solPrimIn.copy()
		self.solCons = solConsIn.copy()

		# add bulk velocity if required
		if (solver.velAdd != 0.0):
			self.solPrim[:,1] += solver.velAdd
			self.solCons, _, _ ,_ = stateFuncs.calcStateFromPrim(self.solPrim, gas)

		# initializing time history
		self.initSolHist(timeInt)

		# snapshot storage matrices, store initial condition
		if solver.primOut: 
			self.primSnap = np.zeros((numCells, gas.numEqs, solver.numSnaps+1), dtype = constants.realType)
			self.primSnap[:,:,0] = solPrimIn.copy()
		if solver.consOut: 
			self.consSnap = np.zeros((numCells, gas.numEqs, solver.numSnaps+1), dtype = constants.realType)
			self.consSnap[:,:,0] = solConsIn.copy()
		if solver.sourceOut: self.sourceSnap = np.zeros((numCells, gas.numSpecies, solver.numSnaps+1), dtype = constants.realType)
		if solver.RHSOut:  self.RHSSnap  = np.zeros((numCells, gas.numEqs, solver.numSnaps), dtype = constants.realType)

		# TODO: initialize thermo properties at initialization too (might be problematic for BC state)

	# initialize time history of solution
	def initSolHist(self, timeIntegrator):
		
		self.solHistCons = [self.solCons.copy()]*(timeIntegrator.timeOrder+1)
		self.solHistPrim = [self.solPrim.copy()]*(timeIntegrator.timeOrder+1)
	
	# update time history of solution for implicit time integration
	def updateSolHist(self):
		self.solHistCons[1:] = self.solHistCons[:-1]
		self.solHistPrim[1:] = self.solHistPrim[:-1]
		self.solHistCons[0] = self.solCons
		self.solHistPrim[0] = self.solPrim
			
	# update solution after a time step 
	# TODO: convert to using class methods
	def updateState(self, gas: gasModel, fromCons = True):

		if fromCons:
			self.solPrim, self.RMix, self.enthRefMix, self.CpMix = stateFuncs.calcStateFromCons(self.solCons, gas)
		else:
			self.solCons, self.RMix, self.enthRefMix, self.CpMix = stateFuncs.calcStateFromPrim(self.solPrim, gas)

	# print residual norms
	# TODO: do some decent formatting on output, depending on resType
	def resOutput(self, solver, tStep):

		dSol = self.solHistPrim[1] - self.solHistPrim[2]
		dSolAbs = np.abs(dSol)

		# L2 norm
		resSumL2 = np.sum(np.square(dSolAbs), axis=0) 		# sum of squares
		resSumL2[:] /= dSol.shape[0]   							# divide by number of cells
		resSumL2 /= np.square(solver.steadyNormPrim) 			# divide by square of normalization constants
		resSumL2 = np.sqrt(resSumL2) 		 					# square root
		resLogL2 = np.log10(resSumL2) 							# get exponent
		resOutL2 = np.mean(resLogL2)
			
		# L1 norm
		resSumL1 = np.sum(dSolAbs, axis=0) 				# sum of absolute values
		resSumL1[:] /= dSol.shape[0]   							# divide by number of cells
		resSumL1 /= solver.steadyNormPrim 						# divide by normalization constants
		resLogL1 = np.log10(resSumL1) 							# get exponent
		resOutL1 = np.mean(resLogL1)

		print(str(tStep+1)+":\tL2 norm: "+str(resOutL2)+",\tL1 norm:"+str(resOutL1))

		self.resOutL2 = resOutL2 
		self.resOutL1 = resOutL1

# grouping class for boundaries
class boundaries:
	
	def __init__(self, sol: solutionPhys, solver):
		# initialize inlet
		pressIn, velIn, tempIn, massFracIn, rhoIn, pertTypeIn, pertPercIn, pertFreqIn = parseBC("inlet", solver.paramDict)
		self.inlet = boundary(solver.paramDict["boundType_inlet"], solver, pressIn, velIn, tempIn, massFracIn, rhoIn,
								pertTypeIn, pertPercIn, pertFreqIn)

		# initialize outlet
		pressOut, velOut, tempOut, massFracOut, rhoOut, pertTypeOut, pertPercOut, pertFreqOut = parseBC("outlet", solver.paramDict)
		self.outlet = boundary(solver.paramDict["boundType_outlet"], solver, pressOut, velOut, tempOut, massFracOut, rhoOut,
								pertTypeOut, pertPercOut, pertFreqOut)


# boundary cell parameters
class boundary:

	def __init__(self, boundType, solver, press, vel, temp, massFrac, rho, pertType, pertPerc, pertFreq):
		
		gas = solver.gasModel

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
		self.sol 		= solutionPhys(solDummy, solDummy, 1, solver)
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
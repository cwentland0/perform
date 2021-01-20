from pygems1d.constants import realType, hugeNum, RUniv

import numpy as np
import pdb


class solutionPhys:
	"""
	Base class for physical solution (opposed to ROM solution)
	"""

	def __init__(self, gas, solPrimIn, numCells):
		
		self.gasModel = gas

		self.numCells = numCells

		# primitive and conservative state
		self.solPrim	= np.zeros((self.gasModel.numEqs, numCells), dtype=realType)		# solution in primitive variables
		self.solCons	= np.zeros((self.gasModel.numEqs, numCells), dtype=realType)		# solution in conservative variables
		
		# chemical properties
		self.mwMix 		= np.zeros(numCells, dtype=realType)								# mixture molecular weight
		self.RMix		= np.zeros(numCells, dtype=realType)								# mixture specific gas constant
		self.gammaMix 	= np.zeros(numCells, dtype=realType)								# mixture ratio of specific heats
		
		# thermodynamic properties
		self.enthRefMix = np.zeros(numCells, dtype=realType)								# mixture reference enthalpy
		self.CpMix 		= np.zeros(numCells, dtype=realType)								# mixture specific heat at constant pressure
		self.h0 		= np.zeros(numCells, dtype=realType) 								# stagnation enthalpy
		self.hi 		= np.zeros((self.gasModel.numSpeciesFull, numCells), dtype=realType)# species enthalpies
		self.c 			= np.zeros(numCells, dtype=realType) 								# sound speed

		# transport properties
		self.dynViscMix   = np.zeros(numCells, dtype=realType)									# mixture dynamic viscosity
		self.thermCondMix = np.zeros(numCells, dtype=realType) 									# mixture thermal conductivity
		self.massDiffMix  = np.zeros((self.gasModel.numSpeciesFull, numCells), dtype=realType) 	# mass diffusivity coefficient (into mixture)

		# derivatives of density and enthalpy
		self.dRho_dP = np.zeros(numCells, dtype=realType)
		self.dRho_dT = np.zeros(numCells, dtype=realType)
		self.dRho_dY = np.zeros((self.gasModel.numSpecies, numCells), dtype=realType)
		self.dH_dP   = np.zeros(numCells, dtype=realType)
		self.dH_dT   = np.zeros(numCells, dtype=realType)
		self.dH_dY   = np.zeros((self.gasModel.numSpecies, numCells), dtype=realType)	

		# reaction rate-of-progress variables
		# TODO: generalize to >1 reaction, reverse reactions
		self.wf = np.zeros((1, numCells), dtype=realType)

		# set initial condition
		assert(solPrimIn.shape == (self.gasModel.numEqs, numCells))
		self.solPrim = solPrimIn.copy()
		self.updateState(fromCons=False)


	def updateState(self, fromCons=True):
		"""
		Update state and some mixture gas properties
		"""

		if fromCons:
			self.calcStateFromCons(calcR=True, calcEnthRef=True, calcCp=True)
		else:
			self.calcStateFromPrim(calcR=True, calcEnthRef=True, calcCp=True)


	def calcStateFromCons(self, calcR=False, calcEnthRef=False, calcCp=False):
		"""
		Compute primitive state from conservative state
		"""

		self.solPrim[3:,:] = self.solCons[3:,:] / self.solCons[[0],:]
		massFracs = self.gasModel.getMassFracArray(solPrim=self.solPrim)

		# update thermo properties
		if calcR:       self.RMix       = self.gasModel.calcMixGasConstant(massFracs)
		if calcEnthRef: self.enthRefMix = self.gasModel.calcMixEnthRef(massFracs)
		if calcCp:      self.CpMix      = self.gasModel.calcMixCp(massFracs)

		# update primitive state
		# TODO: gasModel references
		self.solPrim[1,:] = self.solCons[1,:] / self.solCons[0,:]
		self.solPrim[2,:] = (self.solCons[2,:] / self.solCons[0,:] - np.square(self.solPrim[1,:]) / 2.0 - 
							 self.enthRefMix) / (self.CpMix - self.RMix) 
		self.solPrim[0,:] = self.solCons[0,:] * self.RMix * self.solPrim[2,:]


	def calcStateFromPrim(self, calcR=False, calcEnthRef=False, calcCp=False):
		"""
		Compute state from primitive state
		"""

		massFracs = self.gasModel.getMassFracArray(solPrim=self.solPrim)

		# update thermo properties
		if calcR:       self.RMix       = self.gasModel.calcMixGasConstant(massFracs)
		if calcEnthRef: self.enthRefMix = self.gasModel.calcMixEnthRef(massFracs)
		if calcCp:      self.CpMix      = self.gasModel.calcMixCp(massFracs)

		# update conservative variables
		# TODO: gasModel references
		self.solCons[0,:]  = self.solPrim[0,:] / (self.RMix * self.solPrim[2,:]) 
		self.solCons[1,:]  = self.solCons[0,:] * self.solPrim[1,:]				
		self.solCons[2,:]  = self.solCons[0,:] * ( self.enthRefMix + self.CpMix * self.solPrim[2,:] + 
												 np.power(self.solPrim[1,:], 2.0) / 2.0 ) - self.solPrim[0,:]
		self.solCons[3:,:] = self.solCons[[0],:] * self.solPrim[3:,:]
 

	def calcStateFromRhoH0(self):
		"""
		Adjust pressure and temperature iteratively to agree with a fixed density and stagnation enthalpy
		Used to compute a physically-meaningful Roe average state from the Roe average enthalpy and density
		"""

		densFixed     = np.squeeze(self.solCons[0,:])
		stagEnthFixed = np.squeeze(self.h0)

		dPress = hugeNum * np.ones(self.numCells, dtype=realType)
		dTemp  = hugeNum * np.ones(self.numCells, dtype=realType)

		pressCurr = self.solPrim[0,:]

		iterCount = 0
		onesVec = np.ones(self.numCells, dtype=realType)
		while ( (np.any( np.absolute(dPress / self.solPrim[0,:]) > 0.01 ) or np.any( np.absolute(dTemp / self.solPrim[2,:]) > 0.01)) and (iterCount < 20)):

			# compute density and stagnation enthalpy from current state
			densCurr 		= self.gasModel.calcDensity(self.solPrim)
			stagEnthCurr 	= self.gasModel.calcStagnationEnthalpy(self.solPrim)

			# compute difference between current and fixed density/stagnation enthalpy
			dDens 		= densFixed - densCurr 
			dStagEnth 	= stagEnthFixed - stagEnthCurr

			# compute derivatives of density and stagnation enthalpy with respect to pressure and temperature
			DDensDPress, DDensDTemp = self.gasModel.calcDensityDerivatives(densCurr, wrtPress=True, pressure=self.solPrim[0,:], 
																		   wrtTemp=True, temperature=self.solPrim[2,:])
			DStagEnthDPress, DStagEnthDTemp = self.gasModel.calcStagEnthalpyDerivatives(wrtPress=True, wrtTemp=True, massFracs=self.solPrim[3:,:])

			# compute change in temperature and pressure 
			dFactor = 1.0 / (DDensDPress * DStagEnthDTemp - DDensDTemp * DStagEnthDPress)
			dPress 	= dFactor * (dDens * DStagEnthDTemp - dStagEnth * DDensDTemp)
			dTemp 	= dFactor * (-dDens * DStagEnthDPress + dStagEnth * DDensDPress)

			# threshold change in temperature and pressure 
			dPress  = np.copysign(onesVec, dPress) * np.minimum(np.absolute(dPress), self.solPrim[0,:] * 0.1)
			dTemp 	= np.copysign(onesVec, dTemp) * np.minimum(np.absolute(dTemp), self.solPrim[2,:] * 0.1)

			# update temperature and pressure
			self.solPrim[0,:] += dPress
			self.solPrim[2,:] += dTemp

			iterCount += 1

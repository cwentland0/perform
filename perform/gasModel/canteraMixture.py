from pygems1d.constants import realType, RUniv
from pygems1d.gasModel.gasModel import gasModel

import numpy as np
import cantera as ct
import pdb

class canteraMixture(gasModel):
	"""
	Container class for all Cantera thermo/transport property methods
	"""

	def __init__(self, gasDict, numCells):

		
		self.gas				=	ct.Solution(gasDict["ctiFile"])
		self.gasArray			=	ct.SolutionArray(self.gas,numCells)  #used for keeping track of cell properties
		self.gasChecker			=   ct.SolutionArray(self.gas,numCells)  #used for state independent lookup
		
		self.numSpeciesFull 	= self.gas.n_species				# total number of species in case
		self.SpeciesNames       = self.gas.species_names
		self.molWeights 		= self.gas.molecular_weights	# molecular weights, g/mol
		#These are either not constant or not needed for TPG
		#self.enthRef 			= -2.545e+05#gasDict["enthRef"].astype(realType) 		# reference enthalpy, J/kg
		#self.tempRef 			= 0.0#gasDict["tempRef"]						# reference temperature, K
		#self.Cp 				= gasDict["Cp"].astype(realType)			# heat capacity at constant pressure, J/(kg-K)
		self.Pr 				= gasDict["Pr"].astype(realType)			# Prandtl number
		self.Sc 				= gasDict["Sc"].astype(realType)			# Schmidt number
		self.muRef				= gasDict["muRef"].astype(realType)			# reference dynamic viscosity for Sutherland model
		

		#Don't need these for cantera all reactions are handled internally
		#self.nu 				= gasDict["nu"].astype(realType)		# ?????
		#self.nuArr 				= gasDict["nuArr"].astype(realType)		# ?????
		#self.actEnergy			= float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		#self.preExpFact 		= float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		#self.RGas 				= RUniv / self.molWeights 			# specific gas constant of each species, J/(K*kg)
		
		self.numSpecies 		= self.numSpeciesFull - 1			# last species is not directly solved for
		print(self.numSpecies)
		self.numEqs 			= self.numSpecies + 3				# pressure, velocity, temperature, and species transport
		
		#self.molWeightNu 		= self.molWeights * self.nu 
		#self.mwInvDiffs 		= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		#self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		#self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]

	def setReactors(self,massFracs,temperature,pressure):
		assert(massFracs.shape[0]==self.numSpecies)
		paddedMassFracs = np.concatenate((massFracs, (1 - np.sum(massFracs, axis = 0, keepdims = True))), axis = 0)
		self.gasArray.TPY = temperature,pressure,paddedMassFracs.transpose()
		return 
	def updateReactors(self,massFracs):
		assert(massFracs.shape[0]==self.numSpecies)
		paddedMassFracs = np.concatenate((massFracs, (1 - np.sum(massFracs, axis = 0, keepdims = True))), axis = 0)
		self.gasArray.UVY = None,None,paddedMassFracs.transpose()
		return 
	def calcMixGasConstant(self, solPrim):
		self.setReactors(solPrim[3:,:],solPrim[2,:],solPrim[1,:])
		RMix = RUniv / self.gasArray.mean_molecular_weight
		return RMix

	# compute mixture ratio of specific heats
	def calcMixGamma(self, RMix, CpMix):
		gammaMix = CpMix / (CpMix - RMix)
		return gammaMix


	# REFERNCE ENTHALPY doesn't apply to TPG for now just spitting back mixutre enthalpy
	# compute mixture reference enthalpy 
	def calcMixEnthRef(self, solPrim):
		self.setReactors(solPrim[3:,:],solPrim[2,:],solPrim[1,:])
		enthRefMix=self.gasArray.enthalpy_mass
		return enthRefMix

	# compute mixture specific heat at constant pressure
	def calcMixCp(self, solPrim):
		self.setReactors(solPrim[3:,:],solPrim[2,:],solPrim[1,:])
		return self.gasArray.cp_mass

	# compute density from ideal gas law  
	# TODO: account for compressibility?
	def calcDensity(self, solPrim, RMix=None):

		# need to calculate mixture gas constant
		if (RMix is None):
			massFracs = self.getMassFracArray(solPrim=solPrim)
			RMix = self.calcMixGasConstant(massFracs)

		# calculate directly from ideal gas
		density = solPrim[0,:] / (RMix * solPrim[2,:])

		return density

	# compute individual enthalpies for each species
	def calcSpeciesEnthalpies(self, temperature):

		speciesEnth = self.Cp[:,None] * (np.repeat(np.reshape(temperature, (1,-1)), self.numSpeciesFull, axis=0) - self.tempRef) + self.enthRef[:,None]

		return speciesEnth

	# calculate Denisty from Primitive state
	def calcDensityFromPrim(self,solPrim,solCons):
		RMix = self.calcMixGasConstant(solPrim)
		return solPrim[0,:] / (RMix * solPrim[2,:])
	# calculate Momentum from Primitive state
	def calcMomentumFromPrim(self,solPrim,solCons):
		return solCons[0,:] * solPrim[1,:]
	# calculate Enthalpy from Primitive state
	def calcEnthalpyFromPrim(self,solPrim,solCons):
		self.setReactors(solPrim[3:,:],solPrim[2,:],solPrim[1,:])
		return solCons[0,:] * ( self.gasArray.enthalpy_mass + np.power(solPrim[1,:],2.0) / 2.0 ) - solPrim[0,:]
	# calculate rhoY from Primitive state
	def calcDensityYFromPrim(self,solPrim,solCons):
		return solCons[[0],:]*solPrim[3:,:]

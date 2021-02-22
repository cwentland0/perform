from perform.constants import realType, RUniv, suthTemp
from perform.gasModel.gasModel import gasModel

import numpy as np
import cantera as ct
import pdb

class canteraMixture(gasModel):
	"""
	Container class for all Cantera thermo/transport property methods
	"""

	def __init__(self, gasDict,numCells):

		self.gasType 				= gasDict["gasType"]
		self.gas				=	ct.Solution(gasDict["ctiFile"])
		self.gasArray			=	ct.SolutionArray(self.gas,numCells)  #used for keeping track of cell properties
		#self.gasChecker			=   ct.SolutionArray(self.gas,1)  #used for state independent lookup
		
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
		self.massFracSlice  = np.arange(self.numSpecies)
		#self.molWeightNu 		= self.molWeights * self.nu 
		#self.mwInvDiffs 		= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		#self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		#self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]

	def padMassFrac(self,nm1MassFrac):
		nMassFrac = np.concatenate((nm1MassFrac, (1 - np.sum(nm1MassFrac, axis = 0, keepdims = True))), axis = 0)
		return nMassFrac


	def calcMixGasConstant(self, massFrac):
		self.gasArray.TPY=self.gasArray.T,self.gasArray.P, self.padMassFrac(massFrac).transpose()
		return RUniv/self.gasArray.mean_molecular_weight

	def calcMixGamma(self, RMix, CpMix):
		"""
		Compute mixture ratio of specific heats
		"""
		gammaMix = CpMix / (CpMix - RMix)
		return gammaMix


	def calcEnthalpy(self,density,vel,temp,pressure,Y,enthRefMix,CpMix):
		self.gasArray.TPY=temp,pressure,self.padMassFrac(Y).transpose()
		return density * (self.gasArray.enthalpy_mass + np.power(vel,2.)/2)- pressure


	# compute mixture specific heat at constant pressure
	def calcMixCp(self, massFrac):
		self.gasArray.TPY=self.gasArray.T,self.gasArray.P, self.padMassFrac(massFrac).transpose()
		return self.gasArray.cp_mass

	# compute density from ideal gas law  
	def calcDensity(self, solPrim, RMix=None):
		assert(False)
		return 

	# compute individual enthalpies for each species
	def calcSpeciesEnthalpies(self, temperature):
		assert(False)
		return

	# calculate Denisty from Primitive state
	def calcDensityFromPrim(self,solPrim,solCons):
		assert(False)
		return 
	# calculate Momentum from Primitive state
	def calcMomentumFromPrim(self,solPrim,solCons):
		assert(False)
		return 
	# calculate Enthalpy from Primitive state
	def calcEnthalpyFromPrim(self,solPrim,solCons):
		assert(False)
		return 
	# calculate rhoY from Primitive state
	def calcDensityYFromPrim(self,solPrim,solCons):
		assert(False)
		return 
from constants import realType, RUniv
import numpy as np

# TODO: move stateFuncs concerning gas/transport properties here

class gasModel:

	def __init__(self, gasDict):

		# gas composition
		self.numSpeciesFull 	= int(gasDict["numSpecies"])				# total number of species in case
		self.molWeights 		= gasDict["molWeights"].astype(realType)	# molecular weights, g/mol
		self.enthRef 			= gasDict["enthRef"].astype(realType) 		# reference enthalpy, J/kg
		self.tempRef 			= gasDict["tempRef"]						# reference temperature, K
		self.Cp 				= gasDict["Cp"].astype(realType)			# heat capacity at constant pressure, J/(kg-K)
		self.Pr 				= gasDict["Pr"].astype(realType)			# Prandtl number
		self.Sc 				= gasDict["Sc"].astype(realType)			# Schmidt number
		self.muRef				= gasDict["muRef"].astype(realType)			# reference dynamic viscosity for Sutherland model
		
		# Arrhenius factors
		# TODO: modify these to allow for multiple global reactions
		self.nu 				= gasDict["nu"].astype(realType)		# ?????
		self.nuArr 				= gasDict["nuArr"].astype(realType)		# ?????
		self.actEnergy			= float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		self.preExpFact 		= float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		# TODO: not valid for all gas models
		self.RGas 				= RUniv / self.molWeights 			# specific gas constant, J/(K*kg)
		self.numSpecies 		= self.numSpeciesFull - 1			# last species is not directly solved for
		self.numEqs 			= self.numSpecies + 3				# pressure, velocity, temperature, and species transport
		self.molWeightNu 		= self.molWeights * self.nu 
		self.mwInvDiffs 		= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]

		# mass matrices for calculating viscosity and thermal conductivity mixing laws
		self.mixMassMatrix 		= np.zeros((self.numSpeciesFull, self.numSpeciesFull), dtype=realType)
		self.mixInvMassMatrix 	= np.zeros((self.numSpeciesFull, self.numSpeciesFull), dtype=realType)
		self.precompMixMassMatrices()

	def precompMixMassMatrices(self):
		for specNum in range(self.numSpeciesFull):
			self.mixMassMatrix[specNum, :] 		= np.power((self.molWeights / self.molWeights[specNum]), 0.25)
			self.mixInvMassMatrix[specNum, :] 	= 1.0 / np.sqrt( 1.0 + self.molWeights[specNum] / self.molWeights)

class caloricallyPerfectGas(gasModel):

	def __init__(self, gasDict):
		super().__init__(gasDict)

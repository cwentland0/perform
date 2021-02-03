from perform.constants import realType, RUniv
from perform.inputFuncs import catchInput

import numpy as np
import pdb

# TODO: some of the CPG functions can be generalized and placed here (e.g. calc sound speed in terms of enthalpy and density derivs) 

class gasModel:
	"""
	Base class storing constant chemical properties of modeled species
	Also includes universal gas methods (like calculating mixture molecular weight)
	"""

	def __init__(self, gasDict):

		# gas composition
		self.numSpeciesFull     = int(gasDict["numSpecies"])				# total number of species in case
		self.molWeights         = gasDict["molWeights"].astype(realType)	# molecular weights, g/mol

		# Arrhenius factors
		# TODO: modify these to allow for multiple global reactions
		self.nu                 = gasDict["nu"].astype(realType) 		# global reaction stoichiometric "forward" coefficients
		self.nuArr              = gasDict["nuArr"].astype(realType) 	# global reaction concentration exponents
		self.actEnergy          = float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		self.preExpFact         = float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		self.RGas               = RUniv / self.molWeights 			# specific gas constant of each species, J/(K*kg)	
		self.molWeightNu        = self.molWeights * self.nu 

		# dealing with single-species option
		if (self.numSpeciesFull == 1):
			self.numSpecies	    = self.numSpeciesFull
		else:
			self.numSpecies     = self.numSpeciesFull - 1		# last species is not directly solved for

		self.massFracSlice  = np.arange(self.numSpecies)
		self.mwInv          = 1.0 / self.molWeights
		self.mwInvDiffs     = self.mwInv[self.massFracSlice] - self.mwInv[-1]

		self.numEqs         = self.numSpecies + 3			# pressure, velocity, temperature, and species transport

		# mass matrices for calculating viscosity and thermal conductivity mixing laws
		self.mixMassMatrix 		= np.zeros((self.numSpeciesFull, self.numSpeciesFull), dtype=realType)
		self.mixInvMassMatrix 	= np.zeros((self.numSpeciesFull, self.numSpeciesFull), dtype=realType)
		self.precompMixMassMatrices()

	def precompMixMassMatrices(self):
		"""
		Precompute mass matrices for dynamic viscosity mixing law
		"""

		for specNum in range(self.numSpeciesFull):
			self.mixMassMatrix[specNum, :] 	  = np.power((self.molWeights / self.molWeights[specNum]), 0.25)
			self.mixInvMassMatrix[specNum, :] = (1.0 / (2.0 * np.sqrt(2.0))) * (1.0 / np.sqrt( 1.0 + self.molWeights[specNum] / self.molWeights)) 


	def getMassFracArray(self, solPrim=None, massFracs=None):
		"""
		Helper function to handle array slicing to avoid weird NumPy array broadcasting issues
		"""

		# get all but last mass fraction field
		if (solPrim is None):
			assert (massFracs is not None), "Must provide mass fractions if not providing primitive solution"
			if (massFracs.ndim == 1):
				massFracs = np.reshape(massFracs, (1,-1))

			if (massFracs.shape[1] == self.numSpeciesFull):
				massFracs = massFracs[self.massFracSlice,:]
			else:
				assert (massFracs.shape[0] == self.numSpecies), "If not passing full mass fraction array, must pass N-1 species"
		else:
			massFracs = solPrim[3:,:]

		return massFracs


	def calcAllMassFracs(self, massFracsNS):
		"""
		Helper function to compute all numSpecies_full mass fraction fields from numSpecies fields
		Thresholds all mass fraction fields between zero and unity 
		"""

		if (self.numSpeciesFull == 1):
			massFracs = np.maximum(0.0, np.minimum(1.0, massFracsNS))

		else:
			numSpecies, numCells = massFracsNS.shape
			assert (numSpecies == self.numSpecies), ("massFracsNS argument must have "+str(self.numSpecies)+" species")

			massFracs = np.zeros((numSpecies+1,numCells), dtype=realType)
			massFracs[:-1,:] 	= np.maximum(0.0, np.minimum(1.0, massFracsNS))
			massFracs[-1,:] 	= 1.0 - np.sum(massFracs[:-1,:], axis=0)
			massFracs[-1,:] 	= np.maximum(0.0, np.minimum(1.0, massFracs[-1,:]))

		return massFracs

	def calcMixMolWeight(self, massFracs):
		"""
		Compute mixture molecular weight
		"""

		if (massFracs.shape[0] == self.numSpecies):
			massFracs = self.calcAllMassFracs(massFracs)

		mixMolWeight = 1.0 / np.sum(massFracs / self.molWeights[:, None], axis=0)

		return mixMolWeight


	def calcAllMoleFracs(self, massFracs, mixMolWeight=None):
		"""
		Compute mole fractions of all species from mass fractions
		"""

		if (massFracs.shape[0] == self.numSpecies):
			massFracs = self.calcAllMassFracs(massFracs)

		if (mixMolWeight is None):
			mixMolWeight = self.calcMixMolWeight(massFracs)

		moleFracs = massFracs * mixMolWeight[None, :] * self.mwInv[:, None]

		return moleFracs

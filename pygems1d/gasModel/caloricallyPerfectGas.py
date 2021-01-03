from pygems1d.constants import realType, RUniv
from pygems1d.gasModel.gasModel import gasModel

import numpy as np
import pdb

class caloricallyPerfectGas(gasModel):
	"""
	Container class for all CPG-specific thermo/transport property methods
	"""

	def __init__(self, gasDict):
		super().__init__(gasDict)


	def calcMixGasConstant(self, massFracs):
		if (self.numSpecies > 1):
			RMix = RUniv * ( (1.0 / self.molWeights[-1]) + np.sum(massFracs * self.mwInvDiffs, axis = 1) )
		else:
			RMix = RUniv * ( (1.0 / self.molWeights[-1]) + massFracs * self.mwInvDiffs[0] )
		return RMix

	# compute mixture ratio of specific heats
	def calcMixGamma(self, RMix, CpMix):
		gammaMix = CpMix / (CpMix - RMix)
		return gammaMix

	# compute mixture reference enthalpy
	def calcMixEnthRef(self, massFracs):
		if (self.numSpecies > 1):
			enthRefMix = self.enthRef[-1] + np.sum(massFracs * self.enthRefDiffs, axis = 1)
		else:
			enthRefMix = self.enthRef[-1] + massFracs * self.enthRefDiffs[0]
		return enthRefMix

	# compute mixture specific heat at constant pressure
	def calcMixCp(self, massFracs):
		if (self.numSpecies > 1):
			CpMix = self.Cp[-1] + np.sum(massFracs * self.CpDiffs, axis = 1)
		else:
			CpMix = self.Cp[-1] + massFracs * self.CpDiffs[0]
		return CpMix

	# compute density from ideal gas law
	# TODO: account for compressibility?
	def calcDensity(self, solPrim, RMix=None):

		# need to calculate mixture gas constant
		if (RMix is None):
			massFracs = self.getMassFracArray(solPrim=solPrim)
			RMix = self.calcMixGasConstant(massFracs)

		# calculate directly from ideal gas
		density = solPrim[:,0] / (RMix * solPrim[:,2])

		return density

	# compute individual enthalpies for each species
	def calcSpeciesEnthalpies(self, temperature):

		speciesEnth = self.Cp * (np.repeat(np.reshape(temperature, (-1,1)), 2, axis=1) - self.tempRef) + self.enthRef

		return speciesEnth

	# compute individual dynamic viscosities from Sutherland's law
	# def calcSpeciesDynamicVisc(self, moleFrac=None, massFrac=None, mixMolWeight=None):

	# compute mixture dynamic viscosity from mixing law
	# def calcMixtureDynamicVisc():

	# compute sound speed
	def calcSoundSpeed(self, temperature, RMix=None, gammaMix=None, massFracs=None, CpMix=None):

		# calculate mixture gas constant if not provided
		massFracsSet = False
		if (RMix is None):
			assert (massFracs is not None), "Must provide mass fractions to calculate mixture gas constant..."
			massFracs = self.getMassFracArray(massFracs=massFracs)
			massFracsSet = True
			RMix = self.calcMixGasConstant(massFracs)
		else:
			RMix = np.squeeze(RMix)
			
		# calculate ratio of specific heats if not provided
		if (gammaMix is None):
			if (CpMix is None):
				assert (massFracs is not None), "Must provide mass fractions to calculate mixture Cp..."
				if (not massFracsSet): 
					massFracs = self.getMassFracArray(massFracs=massFracs)
				CpMix = self.calcMixCp(massFracs)
			else:
				CpMix = np.squeeze(CpMix)

			gammaMix = calcMixGamma(RMix, CpMix)
		else:
			gammaMix = np.squeeze(gammaMix)

		soundSpeed = np.sqrt(gammaMix * RMix * temperature)

		return soundSpeed

	# compute derivatives of density with respect to pressure, temperature, or species mass fraction
	def calcDensityDerivatives(self, density, 
								wrtPress=False, pressure=None,
								wrtTemp=False, temperature=None,
								wrtSpec=False, mixMolWeight=None, massFracs=None):

		assert any([wrtPress, wrtTemp, wrtSpec]), "Must compute at least one density derivative..."

		derivs = tuple()
		if (wrtPress):
			assert (pressure is not None), "Must provide pressure for pressure derivative..."
			DDensDPress = density / pressure
			derivs = derivs + (DDensDPress,)

		if (wrtTemp):
			assert (temperature is not None), "Must provide temperature for temperature derivative..."
			DDensDTemp = -density / temperature
			derivs = derivs + (DDensDTemp,)

		if (wrtSpec):
			# calculate mixture molecular weight
			if (mixMolWeight is None):
				assert (massFracs is not None), "Must provide mass fractions to calculate mixture mol weight..."
				mixMolWeight = self.calcMixMolWeight(massFracs)

			DDensDSpec = np.zeros((density.shape[0], self.numSpecies), dtype=realType)
			for specNum in range(self.numSpecies):
				DDensDSpec[:,specNum] = density * mixMolWeight * (1.0 / self.molWeights[-1] - 1.0 / self.molWeights[specNum])
			derivs = derivs + (DDensDSpec,)

		return derivs

	# compute stagnation enthalpy from velocity and species enthalpies
	def calcStagnationEnthalpy(self, solPrim, speciesEnth=None):

		# get the species enthalpies if not provided
		if (speciesEnth is None):
			speciesEnth = self.calcSpeciesEnthalpies(solPrim[:,2])

		# compute all mass fraction fields
		massFracs = self.calcAllMassFracs(solPrim[:,3:])

		# pdb.set_trace()
		assert (massFracs.shape == speciesEnth.shape)

		stagEnth = np.sum(speciesEnth * massFracs, axis=1) + 0.5 * np.square(solPrim[:,1])

		return stagEnth

	# compute derivatives of stagnation enthalpy with respect to pressure, temperature, velocity, or species mass fraction
	def calcStagEnthalpyDerivatives(self, wrtPress=False,
									wrtTemp=False, massFracs=None,
									wrtVel=False, velocity=None,
									wrtSpec=False, speciesEnth=None, temperature=None):

		assert any([wrtPress, wrtTemp, wrtVel, wrtSpec]), "Must compute at least one density derivative..."

		derivs = tuple()
		if (wrtPress):
			DStagEnthDPress = 0.0
			derivs = derivs + (DStagEnthDPress,)
		
		if (wrtTemp):
			assert (massFracs is not None), "Must provide mass fractions for temperature derivative..."

			massFracs = self.getMassFracArray(massFracs=massFracs)
			DStagEnthDTemp = self.calcMixCp(massFracs)
			derivs = derivs + (DStagEnthDTemp,)

		if (wrtVel):
			assert (velocity is not None), "Must provide velocity for velocity derivative..."
			DStagEnthDVel = velocity.copy()
			derivs = derivs + (DStagEnthDVel,)

		if (wrtSpec):
			if (speciesEnth is None):
				assert (temperature is not None), "Must provide temperature if not providing species enthalpies..."
				speciesEnth = self.calcSpeciesEnthalpies(temperature)
			
			DStagEnthDSpec = np.zeros((speciesEnth.shape[0], self.numSpecies), dtype=realType)
			for specNum in range(self.numSpecies):
				DStagEnthDSpec[:,specNum] = speciesEnth[:,-1] - speciesEnth[:,specNum]
			derivs = derivs + (DStagEnthDSpec,)

		return derivs
from perform.constants import realType, RUniv, suthTemp
from perform.gasModel.gasModel import gasModel

import numpy as np
import pdb

# TODO: more options for passing arguments to avoid repeats in called methods

class caloricallyPerfectGas(gasModel):
	"""
	Container class for all CPG-specific thermo/transport property methods
	"""

	def __init__(self, gasDict):
		super().__init__(gasDict)

		self.enthRef = gasDict["enthRef"].astype(realType) 		# reference enthalpy, J/kg
		self.Cp      = gasDict["Cp"].astype(realType)			# heat capacity at constant pressure, J/(kg-K)
		self.Pr      = gasDict["Pr"].astype(realType)			# Prandtl number
		self.Sc      = gasDict["Sc"].astype(realType)			# Schmidt number

		self.muRef   = gasDict["muRef"].astype(realType)		# reference dynamic viscosity for Sutherland model
		self.tempRef = gasDict["tempRef"].astype(realType)		# reference temperature for Sutherland model, K

		self.CpDiffs        = self.Cp[self.massFracSlice] - self.Cp[-1]
		self.enthRefDiffs   = self.enthRef[self.massFracSlice] - self.enthRef[-1]

	def calcMixGasConstant(self, massFracs):
		"""
		Compute mixture specific gas constant
		"""

		massFracsIn = massFracs.copy()
		if (massFracs.shape[0] == self.numSpeciesFull):
			massFracsIn = massFracsIn[self.massFracSlice,:]

		RMix = RUniv * ( (1.0 / self.molWeights[-1]) + np.sum(massFracsIn * self.mwInvDiffs[:,None], axis=0) )
		return RMix


	def calcMixGamma(self, RMix, CpMix):
		"""
		Compute mixture ratio of specific heats
		"""

		gammaMix = CpMix / (CpMix - RMix)
		return gammaMix


	def calcMixEnthRef(self, massFracs):
		"""
		Compute mixture reference enthalpy
		"""

		assert(massFracs.shape[0] == self.numSpecies), "Only numSpecies species must be passed to calcMixEnthRef"
		enthRefMix = self.enthRef[-1] + np.sum(massFracs * self.enthRefDiffs[:,None], axis=0)
		return enthRefMix


	def calcMixCp(self, massFracs):
		"""
		Compute mixture specific heat at constant pressure
		"""

		assert(massFracs.shape[0] == self.numSpecies), "Only numSpecies species must be passed to calcMixCp"
		CpMix = self.Cp[-1] + np.sum(massFracs * self.CpDiffs[:,None], axis=0)
		return CpMix

	
	def calcDensity(self, solPrim, RMix=None):
		"""
		Compute density from ideal gas law
		"""

		# need to calculate mixture gas constant
		if (RMix is None):
			massFracs = self.getMassFracArray(solPrim=solPrim)
			RMix = self.calcMixGasConstant(massFracs)

		# calculate directly from ideal gas
		density = solPrim[0,:] / (RMix * solPrim[2,:])

		return density


	def calcSpeciesEnthalpies(self, temperature):
		"""
		Compute individual enthalpies for each species
		Returns values for ALL species, NOT numSpecies species
		"""

		speciesEnth = self.Cp[:,None] * np.repeat(np.reshape(temperature, (1,-1)), self.numSpeciesFull, axis=0) + self.enthRef[:,None]

		return speciesEnth


	def calcStagnationEnthalpy(self, solPrim, speciesEnth=None):
		"""
		Compute stagnation enthalpy from velocity and species enthalpies
		"""

		# get the species enthalpies if not provided
		if (speciesEnth is None):
			speciesEnth = self.calcSpeciesEnthalpies(solPrim[2,:])

		# compute all mass fraction fields
		massFracs = self.calcAllMassFracs(solPrim[3:,:])

		assert (massFracs.shape == speciesEnth.shape)

		stagEnth = np.sum(speciesEnth * massFracs, axis=0) + 0.5 * np.square(solPrim[1,:])

		return stagEnth


	def calcSpeciesDynamicVisc(self, temperature):
		"""
		Compute individual dynamic viscosities from Sutherland's law
		Defaults to reference dynamic viscosity if reference temperature is zero
		Returns values for ALL species, NOT numSpecies species
		"""

		# TODO: theoretically, I think this should account for species-specific Sutherland temperatures

		specDynVisc = np.zeros((self.numSpeciesFull, len(temperature)), dtype=realType)

		# if reference temperature is (close to) zero, constant dynamic viscosity
		idxsZeroTemp = np.squeeze(np.argwhere(self.tempRef < 1.0e-7), axis=1)
		if (len(idxsZeroTemp) > 0):
			specDynVisc[idxsZeroTemp, :] = self.muRef[idxsZeroTemp, None]

		# otherwise apply Sutherland's law
		idxsSuth = np.squeeze(np.argwhere(self.tempRef > 1.0e-7), axis=1)
		if (len(idxsSuth) > 0):
			tempFac = temperature[None,:] / self.tempRef[idxsSuth, None]
			tempFac = np.power(tempFac, 3./2.)
			suthFac = (self.tempRef[idxsSuth, None] + suthTemp) / (temperature[None, :] + suthTemp)
			specDynVisc[idxsSuth, :] = self.muRef[idxsSuth, None] * tempFac * suthFac

		return specDynVisc


	def calcMixDynamicVisc(self, specDynVisc=None, temperature=None, moleFracs=None, massFracs=None):
		"""
		Compute mixture dynamic viscosity from Wilkes mixing law
		"""

		if (specDynVisc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			specDynVisc = self.calcSpeciesDynamicVisc(temperature)

		if (self.numSpeciesFull == 1):

			mixDynVisc = np.squeeze(specDynVisc)

		else:

			if (moleFracs is None):
				assert (massFracs is not None), "Must provide mass fractions if not providing mole fractions"
				moleFracs = self.calcAllMoleFracs(massFracs)

			phi = np.zeros((self.numSpeciesFull, specDynVisc.shape[1]), dtype=realType)
			for specIdx in range(self.numSpeciesFull):

				muFac = np.sqrt(specDynVisc[[specIdx],:] / specDynVisc)
				phi[specIdx, :] = np.sum(moleFracs * np.square(1.0 + muFac * self.mixMassMatrix[[specIdx],:].T) * self.mixInvMassMatrix[[specIdx],:].T, axis=0)

			mixDynVisc = np.sum( moleFracs * specDynVisc / phi, axis=0)

		return mixDynVisc


	def calcSpeciesThermCond(self, specDynVisc=None, temperature=None):
		"""
		Compute species thermal conductivities
		Returns values for ALL species, NOT numSpecies species
		"""

		if (specDynVisc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			specDynVisc = self.calcSpeciesDynamicVisc(temperature)

		specThermCond = specDynVisc * self.Cp[:, None] / self.Pr[:, None]

		return specThermCond


	def calcMixThermalCond(self, specThermCond=None, specDynVisc=None, temperature=None, moleFracs=None, massFracs=None):
		"""
		Compute mixture thermal conductivity
		"""

		if (specThermCond is None):
			assert ((specDynVisc is not None) or (temperature is not None)), \
					"Must provide species dynamic viscosity or temperature if not providing species thermal conductivity"
			specThermCond = self.calcSpeciesThermCond(specDynVisc=specDynVisc, temperature=temperature)

		if (self.numSpeciesFull == 1):

			mixThermCond = np.squeeze(specThermCond)

		else:

			if (moleFracs is None):
				assert (massFracs is not None), "Must provide mass fractions if not providing mole fractions"
				moleFracs = self.calcAllMoleFracs(massFracs)

			mixThermCond = 0.5 * ( np.sum(moleFracs * specThermCond, axis=0) + 1.0 / np.sum(moleFracs / specThermCond, axis=0) )

		return mixThermCond


	def calcSpeciesMassDiffCoeff(self, density, specDynVisc=None, temperature=None):
		"""
		Compute mass diffusivity coefficient of species into mixture
		Returns values for ALL species, NOT numSpecies species
		"""

		if (specDynVisc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			specDynVisc = self.calcSpeciesDynamicVisc(temperature)
		
		specMassDiff = specDynVisc / (self.Sc[:, None] * density[None, :])

		return specMassDiff


	def calcSoundSpeed(self, temperature, RMix=None, gammaMix=None, massFracs=None, CpMix=None):
		"""
		Compute sound speed
		"""

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

			gammaMix = self.calcMixGamma(RMix, CpMix)
		else:
			gammaMix = np.squeeze(gammaMix)

		soundSpeed = np.sqrt(gammaMix * RMix * temperature)

		return soundSpeed


	def calcDensityDerivatives(self, density, 
								wrtPress=False, pressure=None,
								wrtTemp=False, temperature=None,
								wrtSpec=False, mixMolWeight=None, massFracs=None):

		"""
		Compute derivatives of density with respect to pressure, temperature, or species mass fraction
		For species derivatives, returns numSpecies derivatives
		"""

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

			DDensDSpec = np.zeros((self.numSpecies, density.shape[0]), dtype=realType)
			for specNum in range(self.numSpecies):
				DDensDSpec[specNum, :] = density * mixMolWeight * (1.0 / self.molWeights[-1] - 1.0 / self.molWeights[specNum])
			derivs = derivs + (DDensDSpec,)

		return derivs


	def calcStagEnthalpyDerivatives(self, wrtPress=False,
									wrtTemp=False, massFracs=None,
									wrtVel=False, velocity=None,
									wrtSpec=False, speciesEnth=None, temperature=None):

		"""
		Compute derivatives of stagnation enthalpy with respect to pressure, temperature, velocity, or species mass fraction
		For species derivatives, returns numSpecies derivatives
		"""

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
			
			DStagEnthDSpec = np.zeros((self.numSpecies, speciesEnth.shape[1]), dtype=realType)
			if (self.numSpeciesFull == 1):
				DStagEnthDSpec[0,:] = speciesEnth[0,:]
			else:
				for specNum in range(self.numSpecies):
					DStagEnthDSpec[specNum,:] = speciesEnth[specNum,:] - speciesEnth[-1,:]

			derivs = derivs + (DStagEnthDSpec,)

		return derivs
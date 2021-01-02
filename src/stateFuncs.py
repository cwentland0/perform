import numpy as np
from gasModel import gasModel
import constants
from constants import realType, RUniv
import pdb

# TODO: could possibly convert these to solution methods
# TODO: these functions are unreasonably slow. Thermo property routines are especially slow, even for ghost cells

# compute primitive variables from conservative variables and mixture thermo properties
def calcStateFromCons(solCons, gas: gasModel):

	# pressure, velocity, temperature, mass fraction
	solPrim = np.zeros(solCons.shape, dtype = realType)

	solPrim[:,3:] 	= solCons[:,3:] / solCons[:,[0]] 
	massFracs = getMassFracArray(gas, solPrim=solPrim)

	# update thermo properties
	RMix 			= calcGasConstantMixture(massFracs, gas)
	enthRefMix 		= calcEnthRefMixture(massFracs, gas)
	CpMix 			= calcCpMixture(massFracs, gas)

	# update primitive state
	solPrim[:,1] = solCons[:,1] / solCons[:,0]
	solPrim[:,2] = (solCons[:,2] / solCons[:,0] - np.square(solPrim[:,1])/2.0 - enthRefMix + CpMix * gas.tempRef) / (CpMix - RMix) 
	solPrim[:,0] = solCons[:,0] * RMix * solPrim[:,2]
	
	return solPrim, RMix, enthRefMix, CpMix

# compute conservative variables from primitive variables and mixture thermo properties
def calcStateFromPrim(solPrim, gas: gasModel):

	# density, momentum, energy, density-weighted mass fraction
	solCons = np.zeros(solPrim.shape, dtype = realType)

	if (solPrim.dtype==constants.complexType): #for complex step jacobian
		solCons = np.zeros(solPrim.shape, dtype = constants.complexType)

	massFracs = getMassFracArray(gas, solPrim=solPrim)

	# update thermo properties
	RMix 			= calcGasConstantMixture(massFracs, gas)
	enthRefMix 		= calcEnthRefMixture(massFracs, gas)
	CpMix 			= calcCpMixture(massFracs, gas)

	# update conservative variables
	solCons[:,0] = solPrim[:,0] / (RMix * solPrim[:,2]) 
	solCons[:,1] = solCons[:,0] * solPrim[:,1]				
	solCons[:,2] = solCons[:,0] * ( enthRefMix + CpMix * (solPrim[:,2] - gas.tempRef) + np.power(solPrim[:,1],2.0) / 2.0 ) - solPrim[:,0]
	solCons[:,3:] = solCons[:,[0]]*solPrim[:,3:]

	return solCons, RMix, enthRefMix, CpMix

# TODO: faster implementation of these in PTM

# compute mixture specific gas constant
def calcGasConstantMixture(massFracs, gas: gasModel):
	if (gas.numSpecies > 1):
		RMix = RUniv * ( (1.0 / gas.molWeights[-1]) + np.sum(massFracs * gas.mwInvDiffs, axis = 1) )
	else:
		RMix = RUniv * ( (1.0 / gas.molWeights[-1]) + massFracs * gas.mwInvDiffs[0] )
	return RMix

# compute mixture ratio of specific heats
def calcGammaMixture(RMix, CpMix):
	gammaMix = CpMix / (CpMix - RMix)
	return gammaMix

# compute mixture reference enthalpy
def calcEnthRefMixture(massFracs, gas: gasModel):
	if (gas.numSpecies > 1):
		enthRefMix = gas.enthRef[-1] + np.sum(massFracs * gas.enthRefDiffs, axis = 1)
	else:
		enthRefMix = gas.enthRef[-1] + massFracs * gas.enthRefDiffs[0]
	return enthRefMix

# compute mixture specific heat at constant pressure
def calcCpMixture(massFracs, gas: gasModel):
	if (gas.numSpecies > 1):
		CpMix = gas.Cp[-1] + np.sum(massFracs * gas.CpDiffs, axis = 1)
	else:
		CpMix = gas.Cp[-1] + massFracs * gas.CpDiffs[0]
	return CpMix

# array slicing to avoid weird NumPy array broadcasting issues
def getMassFracArray(gas: gasModel, solPrim=None, massFracs=None):

	# get all but last mass fraction field
	if (solPrim is None):
		assert (massFracs is not None), "Must provide mass fractions if not providing primitive solution"
		if (massFracs.ndim == 1):
			massFracs = np.reshape(massFracs, (-1,1))
		if (massFracs.shape[1] == gas.numSpeciesFull):
			massFracs = massFracs[:,:-1]
	else:
		massFracs = solPrim[:,3:]

	# slice array appropriately
	if (gas.numSpecies > 1):
		massFracs = massFracs[:,:-1]
	else:
		massFracs = massFracs[:,0]

	return massFracs

# compute all numSpecies_full mass fraction fields from numSpecies fields
# thresholds all mass fraction fields between zero and unity 
def calcAllMassFracs(massFracsNS):

	numCells, numSpecies = massFracsNS.shape
	massFracs = np.zeros((numCells, numSpecies+1), dtype=constants.realType)

	massFracs[:, :-1] 	= np.maximum(0.0, np.minimum(1.0, massFracsNS))
	massFracs[:, -1] 	= 1.0 - np.sum(massFracs[:, :-1], axis=1)
	massFracs[:, -1] 	= np.maximum(0.0, np.minimum(1.0, massFracs[:, -1]))

	return massFracs

# compute density from ideal gas law
def calcDensity(solPrim, gas: gasModel, RMix=None):

	# need to calculate mixture gas constant
	if (RMix is None):
		massFracs = getMassFracArray(gas, solPrim=solPrim)
		RMix = calcGasConstantMixture(massFracs, gas)

	# calculate directly from ideal gas
	density = solPrim[:,0] / (RMix * solPrim[:,2])

	return density

# compute individual enthalpies for each species
def calcSpeciesEnthalpies(temperature, gas: gasModel):

	speciesEnth = gas.Cp * (np.repeat(np.reshape(temperature, (-1,1)), 2, axis=1) - gas.tempRef) + gas.enthRef

	return speciesEnth

# compute stagnation enthalpy from velocity and species enthalpies
def calcStagnationEnthalpy(solPrim, gas: gasModel, speciesEnth=None):

	# get the species enthalpies if not provided
	if (speciesEnth is None):
		speciesEnth = calcSpeciesEnthalpies(solPrim[:,2], gas)

	# compute all mass fraction fields
	massFracs = calcAllMassFracs(solPrim[:,3:])

	assert (massFracs.shape == speciesEnth.shape)

	stagEnth = np.sum(speciesEnth * massFracs, axis=1) + 0.5 * np.square(solPrim[:,1])

	return stagEnth

# compute mixture molecular weight
def calcMolWeightMixture(massFracs, gas: gasModel):

	if (massFracs.shape[1] == gas.numSpecies):
		massFracs = calcAllMassFracs(massFracs)

	mixMolWeight = 1.0 / np.sum(massFracs / gas.molWeights, axis=1)

	return mixMolWeight

# compute individual dynamic viscosities from Sutherland's law
# def calcSpeciesDynamicVisc(gas: gasModel, moleFrac=None, massFrac=None, mixMolWeight=None):

# compute mixture dynamic viscosity from mixing law
# def calcMixtureDynamicVisc():

# compute sound speed
def calcSoundSpeed(temperature, RMix=None, gammaMix=None, gas: gasModel=None, massFracs=None, CpMix=None):

	# calculate mixture gas constant if not provided
	massFracsSet = False
	if (RMix is None):
		assert (gas is not None), "Must provide gas properties to calculate mixture gas constant..."
		assert (massFracs is not None), "Must provide mass fractions to calculate mixture gas constant..."
		massFracs = getMassFracArray(gas, massFracs=massFracs)
		massFracsSet = True
		RMix = calcGasConstantMixture(massFracs, gas)
	else:
		RMix = np.squeeze(RMix)
		
	# calculate ratio of specific heats if not provided
	if (gammaMix is None):
		if (CpMix is None):
			assert (massFracs is not None), "Must provide mass fractions to calculate mixture Cp..."
			if (not massFracsSet): 
				massFracs = getMassFracArray(gas, massFracs=massFracs)
			CpMix = calcCpMixture(massFracs, gas)
		else:
			CpMix = np.squeeze(CpMix)

		gammaMix = calcGammaMixture(RMix, CpMix)
	else:
		gammaMix = np.squeeze(gammaMix)

	soundSpeed = np.sqrt(gammaMix * RMix * temperature)

	return soundSpeed

# compute derivatives of density with respect to pressure, temperature, or species mass fraction
def calcDensityDerivatives(density, 
							wrtPress=False, pressure=None,
							wrtTemp=False, temperature=None,
							wrtSpec=False, mixMolWeight=None, gas: gasModel=None, massFracs=None):

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
		assert (gas is not None), "Must provide gas properties for species derivative..."
		if (mixMolWeight is None):
			assert (massFracs is not None), "Must provide mass fractions to calculate mixture mol weight..."
			mixMolWeight = calcMolWeightMixture(massFracs, gas)

		DDensDSpec = np.zeros((density.shape[0], gas.numSpecies), dtype=constants.realType)
		for specNum in range(gas.numSpecies):
			DDensDSpec[:,specNum] = density * mixMolWeight * (1.0/gas.molWeights[-1] - 1.0/gas.molWeights[specNum])
		derivs = derivs + (DDensDSpec,)

	return derivs

# compute derivatives of stagnation enthalpy with respect to pressure, temperature, velocity, or species mass fraction
def calcStagEnthalpyDerivatives(wrtPress=False,
								wrtTemp=False, massFracs=None, gas:gasModel=None,
								wrtVel=False, velocity=None,
								wrtSpec=False, speciesEnth=None, temperature=None):

	assert any([wrtPress, wrtTemp, wrtVel, wrtSpec]), "Must compute at least one density derivative..."

	derivs = tuple()
	if (wrtPress):
		DStagEnthDPress = 0.0
		derivs = derivs + (DStagEnthDPress,)
	
	if (wrtTemp):
		assert ((massFracs is not None) and (gas is not None)), "Must provide mass fractions and gas properties for temperature derivative..."

		massFracs = getMassFracArray(gas, massFracs=massFracs)
		DStagEnthDTemp = calcCpMixture(massFracs, gas)
		derivs = derivs + (DStagEnthDTemp,)

	if (wrtVel):
		assert (velocity is not None), "Must provide velocity for velocity derivative..."
		DStagEnthDVel = velocity.copy()
		derivs = derivs + (DStagEnthDVel,)

	if (wrtSpec):
		if (speciesEnth is None):
			assert (temperature is not None), "Must provide temperature if not providing species enthalpies..."
			speciesEnth = calcSpeciesEnthalpies(temperature, gas)
		
		DStagEnthDSpec = np.zeros((speciesEnth.shape[0], gas.numSpecies), dtype=constants.realType)
		for specNum in range(gas.numSpecies):
			DStagEnthDSpec[:,specNum] = speciesEnth[:,-1] - speciesEnth[:,specNum]
		derivs = derivs + (DStagEnthDSpec,)

	return derivs

# Adjust pressure and temperature iterative to agree with a fixed density and stagnation enthalpy
# Used to compute a physically-meaningful Roe average state from the Roe average enthalpy and density 
def calcStateFromRhoH0(solPrim, densFixed, stagEnthFixed, gas: gasModel):

	densFixed 		= np.squeeze(densFixed)
	stagEnthFixed 	= np.squeeze(stagEnthFixed)

	dPress 		= constants.hugeNum * np.ones(solPrim.shape[0], dtype=np.float64)
	dTemp 		= constants.hugeNum * np.ones(solPrim.shape[0], dtype=np.float64)

	pressCurr 		= solPrim[:,0]

	iterCount = 0
	onesVec = np.ones(solPrim.shape[0], dtype=constants.realType)
	while ( (np.any( np.absolute(dPress / solPrim[:,0]) > 0.01 ) or np.any( np.absolute(dTemp / solPrim[:,2]) > 0.01)) and (iterCount < 20)):

		# compute density and stagnation enthalpy from current state
		densCurr 		= calcDensity(solPrim, gas)
		stagEnthCurr 	= calcStagnationEnthalpy(solPrim, gas)

		# compute difference between current and fixed density/stagnation enthalpy
		dDens 		= densFixed - densCurr 
		dStagEnth 	= stagEnthFixed - stagEnthCurr

		# compute derivatives of density and stagnation enthalpy with respect to pressure and temperature
		DDensDPress, DDensDTemp = calcDensityDerivatives(densCurr, wrtPress=True, pressure=solPrim[:,0], wrtTemp=True, temperature=solPrim[:,2])
		DStagEnthDPress, DStagEnthDTemp = calcStagEnthalpyDerivatives(wrtPress=True, wrtTemp=True, massFracs=solPrim[:,3:], gas=gas)

		# compute change in temperature and pressure 
		dFactor = 1.0 / (DDensDPress * DStagEnthDTemp - DDensDTemp * DStagEnthDPress)
		dPress 	= dFactor * (dDens * DStagEnthDTemp - dStagEnth * DDensDTemp)
		dTemp 	= dFactor * (-dDens * DStagEnthDPress + dStagEnth * DDensDPress)

		# threshold change in temperature and pressure 
		dPress  = np.copysign(onesVec, dPress) * np.minimum(np.absolute(dPress), solPrim[:,0] * 0.1)
		dTemp 	= np.copysign(onesVec, dTemp) * np.minimum(np.absolute(dTemp), solPrim[:,2] * 0.1)

		# update temperature and pressure
		solPrim[:,0] += dPress
		solPrim[:,2] += dTemp

		iterCount += 1

	return solPrim
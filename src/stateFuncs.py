import numpy as np
from classDefs import gasProps
import constants
from constants import realType, RUniv
import pdb

# TODO: could possibly convert these to solution methods
# TODO: these functions are unreasonably slow. Thermo property routines are especially slow, even for ghost cells

# compute primitive variables from conservative variables and mixture thermo properties
# @profile
def calcStateFromCons(solCons, gas: gasProps):

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
# @profile
def calcStateFromPrim(solPrim, gas: gasProps):

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
def calcGasConstantMixture(massFracs, gas: gasProps):
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
def calcEnthRefMixture(massFracs, gas: gasProps):
	if (gas.numSpecies > 1):
		enthRefMix = gas.enthRef[-1] + np.sum(massFracs * gas.enthRefDiffs, axis = 1)
	else:
		enthRefMix = gas.enthRef[-1] + massFracs * gas.enthRefDiffs[0]
	return enthRefMix

# compute mixture specific heat at constant pressure
def calcCpMixture(massFracs, gas: gasProps):
	if (gas.numSpecies > 1):
		CpMix = gas.Cp[-1] + np.sum(massFracs * gas.CpDiffs, axis = 1)
	else:
		CpMix = gas.Cp[-1] + massFracs * gas.CpDiffs[0]
	return CpMix

# array slicing to avoid weird NumPy array broadcasting issues
def getMassFracArray(gas: gasProps, solPrim=None, massFracs=None):

	# get all but last mass fraction field
	if (solPrim is None):
		assert(massFracs is not None), "Must provide mass fractions if not providing primitive solution"
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
def calcDensity(solPrim, gas: gasProps, RMix=None):

	# need to calculate mixture gas constant
	if (RMix is None):
		massFracs = getMassFracArray(gas, solPrim=solPrim)
		RMix = calcGasConstantMixture(massFracs, gas)

	# calculate directly from ideal gas
	density = solPrim[:,0] / (RMix * solPrim[:,2])

	return density

# compute individual enthalpies for each species
def calcSpeciesEnthalpies(temperature, gas: gasProps):

	speciesEnth = gas.Cp * (np.repeat(np.reshape(temperature, (-1,1)), 2, axis=1) - gas.tempRef) + gas.enthRef

	return speciesEnth

# compute stagnation enthalpy from velocity and species enthalpies
def calcStagnationEnthalpy(solPrim, gas: gasProps, speciesEnth=None):

	# get the species enthalpies if not provided
	if (speciesEnth is None):
		speciesEnth = calcSpeciesEnthalpies(solPrim[:,2], gas)

	# compute all mass fraction fields
	massFracs = calcAllMassFracs(solPrim[:,3:])

	assert (massFracs.shape == speciesEnth.shape)

	stagEnth = np.sum(speciesEnth * massFracs, axis=1) + 0.5 * np.square(solPrim[:,1])

	return stagEnth

# compute mixture molecular weight
def calcMolWeightMixture(massFracs, gas: gasProps):

	if (massFracs.shape[1] == gas.numSpecies):
		massFracs = calcAllMassFracs(massFracs)

	mixMolWeight = 1.0 / np.sum(massFracs / gas.molWeights, axis=1)

	return mixMolWeight

# compute individual dynamic viscosities from Sutherland's law
# def calcSpeciesDynamicVisc(gas: gasProps, moleFrac=None, massFrac=None, mixMolWeight=None):

# compute mixture dynamic viscosity from mixing law
# def calcMixtureDynamicVisc():

# compute sound speed
def calcSoundSpeed(temperature, RMix=None, gammaMix=None, gas: gasProps=None, massFracs=None, CpMix=None):

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
							wrtSpec=False, mixMolWeight=None, gas: gasProps=None, massFracs=None):

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
								wrtTemp=False, massFracs=None, gas:gasProps=None,
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
def calcStateFromRhoH0(solPrim, densFixed, stagEnthFixed, gas: gasProps):

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

##### state functions ripped directly from ptm.f90 ####
# TODO: there are lot of redundant, repeated operations in many cases. Not sure if it can be cleaned up
# TODO: rename, comment these with more explanatory text
# TODO: modify these to work for full state vectors, not just single cells 
# expand these beyond perfect gases

def calcStateEigenvalues(solPrim, gas: gasProps):

	vel 	= solPrim[1]
	rhox 	= calcRhoxf(solPrim, gas)
	h0x 	= calcH0xf(solPrim, gas)
	rho 	= calcRhof(solPrim, gas)
	
	rhop 	= rhox[0]
	rhot 	= rhox[2]
	hp 		= h0x[0]
	ht 		= h0x[2]
	rhopp 	= rhop 

	dpp 	= rhopp + rhot * (1.0 - rho * hp) / (rho * ht)

	if (abs(dpp) <= constants.tinyNum):
		raise ValueError("Error in state eigenvalue routine")

	dpp 	= 1.0 / dpp
	rhopmpp = (rhop - rhopp) * dpp 
	velb 	= 0.5 * rhopmpp * vel 
	dpp 	= np.sqrt(abs(vel * vel + dpp))

	c1 		= vel + velb + dpp
	c2 		= vel + velb - dpp

	return c1, c2

def calcStateEigenvectors(solPrim, c1, c2, gas: gasProps):

	rhox 	= calcRhoxf(solPrim, gas)
	h0x 	= calcH0xf(solPrim, gas)
	rho 	= calcRhof(solPrim, gas)
	
	hp 		= h0x[0]
	ht		= h0x[2]
	vel 	= solPrim[1]

	htinv 	= (1.0 - rho * hp) / ht
	c1mvel 	= c1 - vel
	c2mvel 	= c2 - vel 
	c2mc1inv 		= 1.0 / (c2 - c1)
	rhoc2mc1inv 	= c2mc1inv / rho 
	c1mveloc2mc1 	= c1mvel * c2mc1inv
	c2mveloc2mc1 	= c2mvel * c2mc1inv

	zm = np.zeros((gas.numEqs, gas.numEqs), dtype=realType)
	zmInv = np.zeros((gas.numEqs, gas.numEqs), dtype=realType)

	zm[0,1] = c1mvel * rho 
	zm[0,2] = c2mvel * rho
	zm[1,1] = 1.0
	zm[1,2] = 1.0
	zm[2,0] = 1.0
	zm[2,1] = c1mvel * htinv
	zm[2,2] = c2mvel * htinv

	zmInv[0,0] = -htinv / rho
	zmInv[0,3] = 1.0
	zmInv[1,0] = -rhoc2mc1inv
	zmInv[1,1] = c2mveloc2mc1
	zmInv[2,0] = rhoc2mc1inv
	zmInv[2,1] = -c1mveloc2mc1

	for specIdx in range(3,gas.numEqs+1):
		zm[specIdx,specIdx] 	= 1.0
		zmInv[specIdx,specIdx] 	= 1.0

	return zm, zmInv


def calcRhoxf(solPrim, gas: gasProps):

	yi 		= calcYif(solPrim, gas)
	rhoi 	= calcRhoif(solPrim, gas) 
	rhopi 	= calcRhopi(solPrim, gas)
	rhoti 	= calcRhoti(solPrim, gas)

	wrho 	= rho / rhoi
	wrho 	= wrho * wrho * yi
	rhot 	= np.sum(wrho * rhoti)
	rhop 	= np.sum(wrho * rhopi)

	rhoxf = np.zeros(gas.numEqs, dtype=realType)
	rhoxf[0] = rhop
	rhoxf[2] = rhot
	rhoxf[3:] = - rho * rho * (1.0 / rhoi[:-1] - 1.0 / rhoi[-1])

	return rhoxf


def calcRhof(solPrim, gas: gasProps):

	rhoi 	= calcRhoif(solPrim, gas)
	yi 		= calcYif(solPrim, gas)
	rhof 	= 1.0 / np.sum(yi / rhoi)

	return rhof

def calcRhoif(solPrim, gas: gasProps):

	# perfect gas
	press 	= solPrim[0]
	temp 	= solPrim[2]
	rhoif 	= press / (gas.RGas * temp)

	return rhoif

# derivative of density with respect to pressure
def calcRhopif(solPrim, gas: gasProps):

	# perfect gas 
	temp 	= solPrim[2]
	rhopif 	= 1.0 / (gas.RGas * temp)

	return rhopif

# derivative of density with respect to temperature
def calcRhotif(solPrim, gas: gasProps):

	# perfect gas
	press 	= solPrim[0]
	temp 	= solPrim[2]
	rhotif 	= - press / (gas.RGas * temp**2)

	return rhotif

# threshold pressure to small number, at minimum
def calcPf(solPrim, gas: gasProps):

	pf = max(solPrim[0], constants.tinyNum)
	return pf


def calcH0xf(solPrim, gas: gasProps):

	h0xf = np.zeros(gas.numEqs, dtype=realType)
	h0xf[0] = calcHpf(solPrim, gas)
	h0xf[1] = solPrim[1]
	h0xf[2] = calcHtf(solPrim, gas)

	hi = calcHif(solPrim, gas)
	h0xf[3:] = hi[:-1] - hi[-1]

	return h0xf

# mass fraction weighted sum of derivatives of enthalpy with respect to pressure
def calcHpf(solPrim, gas: gasProps):

	hpi = calcHpif(solPrim, gas)
	yi 	= calcYif(solPrim, gas)
	hpf = np.sum(hpi * yi)

	return hpf

# mass fraction weighted sum of derivatives of enthalpy with respect to temperature
def calcHtf(solPrim, gas: gasProps):

	hti = calcHtif(solPrim, gas)
	yi 	= calcYif(solPrim, gas)
	htf = np.sum(hti * yi)

	return htf

# I have no idea what the original intended purpose of this function is
def calcHzf(solPrim, gas: gasProps):

	return constants.q0

def hif(solPrim, gas: gasProps):

	hif = np.zeros(gas.numSpeciesFull, dtype=realType)

	# perfect gas
	hif = gas.enthRef + gas.Cp * (solPrim[2] - constants.enthRefTemp)

	return hif

# derivative of enthalpy with respect to pressure
def calcHpif(solPrim, gas: gasProps):

	hpif = np.zeros(gas.numSpeciesFull, dtype=realType)

	# perfect gas
	hpif = 0.0

	return hpif

# derivative of enthalpy with respect to temperature
def calcHtif(solPrim, gas: gasProps):

	htif = np.zeros(gas.numSpeciesFull, dtype=realType)

	# perfect gas
	htif = gas.Cp

	return htif

# threshold species mass fractions between 0 and 1
def calcYif(solPrim, gas: gasProps):

	yif = np.zeros(gas.numSpeciesFull, dtype=realType)
	
	yif[:-1] = np.amin(np.amax(0.0, solPrim[3:]), 1.0)
	yif[-1] = 1.0 - np.sum(yif[:-1])

	return yif
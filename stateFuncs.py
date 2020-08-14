import numpy as np
from classDefs import gasProps
import constants
import pdb

# TODO: could possibly convert these to solution methods
# TODO: these functions are unreasonably slow. Thermo property routines are especially slow, even for ghost cells

# compute primitive variables from conservative variables and mixture thermo properties
# @profile
def calcStateFromCons(solCons, gas: gasProps):

	# pressure, velocity, temperature, mass fraction
	solPrim = np.zeros(solCons.shape, dtype = constants.floatType)

	solPrim[:,3:] 	= solCons[:,3:] / solCons[:,[0]] 
	if (gas.numSpecies > 1):
		massFracs = solPrim[:,3:]
	else:
		massFracs = solPrim[:,3]

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
	solCons = np.zeros(solPrim.shape, dtype = constants.floatType)

	if (solPrim.dtype==np.complex64): #for complex step jacobian
		solCons = np.zeros(solPrim.shape, dtype = np.complex64)

	if (gas.numSpecies > 1):
		massFracs = solPrim[:,3:]
	else:
		massFracs = solPrim[:,3]

	# update thermo properties
	RMix 			= calcGasConstantMixture(massFracs, gas)
	enthRefMix 		= calcEnthRefMixture(massFracs, gas)
	CpMix 			= calcCpMixture(massFracs, gas)

	# update conservative variables
	solCons[:,0] = solPrim[:,0] / (RMix * solPrim[:,2]) 
	solCons[:,1] = solCons[:,0] * solPrim[:,1]				
	solCons[:,2] = solCons[:,0] * ( enthRefMix + CpMix * (solPrim[:,2] - gas.tempRef) + np.power(solPrim[:,1],2.0)/2.0 ) - solPrim[:,0]
	solCons[:,3:] = solCons[:,[0]]*solPrim[:,3:]

	return solCons, RMix, enthRefMix, CpMix

# compute mixture specific gas constant
def calcGasConstantMixture(massFracs, gas: gasProps):
	if (gas.numSpecies > 1):
		RMix = constants.RUniv * ( (1.0 / gas.molWeights[-1]) + np.sum(massFracs * gas.mwDiffs, axis = 1) )
	else:
		RMix = constants.RUniv * ( (1.0 / gas.molWeights[-1]) + massFracs * gas.mwDiffs[0] )
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
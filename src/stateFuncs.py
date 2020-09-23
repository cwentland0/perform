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
	solCons = np.zeros(solPrim.shape, dtype = realType)

	if (solPrim.dtype==constants.complexType): #for complex step jacobian
		solCons = np.zeros(solPrim.shape, dtype = constants.complexType)

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
	solCons[:,2] = solCons[:,0] * ( enthRefMix + CpMix * (solPrim[:,2] - gas.tempRef) + np.power(solPrim[:,1],2.0) / 2.0 ) - solPrim[:,0]
	solCons[:,3:] = solCons[:,[0]]*solPrim[:,3:]

	return solCons, RMix, enthRefMix, CpMix

# TODO: faster implementation of these in PTM

# compute mixture specific gas constant
def calcGasConstantMixture(massFracs, gas: gasProps):
	if (gas.numSpecies > 1):
		RMix = RUniv * ( (1.0 / gas.molWeights[-1]) + np.sum(massFracs * gas.mwDiffs, axis = 1) )
	else:
		RMix = RUniv * ( (1.0 / gas.molWeights[-1]) + massFracs * gas.mwDiffs[0] )
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
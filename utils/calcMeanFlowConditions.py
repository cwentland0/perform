from perform.solution.solutionPhys import solutionPhys
from perform.inputFuncs import readInputFile
from perform.gasModel.caloricallyPerfectGas import caloricallyPerfectGas

import numpy as np
from math import sqrt
import os
import pdb

##### BEGIN USER INPUT #####

gasFile = "~/path/to/chemistry/file.chem"

# If True, load primitive state from icFile. If False, specify left and right primitive state
fromICFile = False  
icFile = ""

# left and right states, if fomrICFile = False
pressL    = 1e5
velL      = 0.0
tempL     = 256.420677
massFracL = [1.0]
pressR    = 1e4
velR      = 0.0
tempR     = 256.420677
massFracR = [1.0]

##### END USER INPUT #####

gasFile = os.path.expanduser(gasFile)

# load gas file
gasDict = readInputFile(gasFile)
gasType = gasDict["gasType"]
if (gasType == "cpg"):
	gas = caloricallyPerfectGas(gasDict)
else:
	raise ValueError("Invalid gasType")

# handle single-species
numSpeciesFull = len(massFracL)
assert (len(massFracR) == numSpeciesFull), "massFracL and massFracR must have the same number of mass fractions"
assert (np.sum(massFracL) == 1.0), "massFracL elements must sum to 1.0"
assert (np.sum(massFracR) == 1.0), "massFracR elements must sum to 1.0"
if (numSpeciesFull == 1):
	numSpecies = numSpeciesFull
else:
	numSpecies = numSpeciesFull - 1
massFracSlice = np.arange(numSpecies)
 
# set up states
if fromICFile:
	solPrim    = np.load(icFile)
	solPrimIn  = solPrim[:, [0]]
	solPrimOut = solPrim[:, [-1]]

else:
	solPrimIn        = np.zeros((3+numSpecies,1), dtype = np.float64)
	solPrimOut       = np.zeros((3+numSpecies,1), dtype = np.float64)
	solPrimIn[:3,0]  = np.array([pressL, velL, tempL])
	solPrimIn[3:,0]  = np.array(massFracL).astype(np.float64)[massFracSlice]
	solPrimOut[:3,0] = np.array([pressR, velR, tempR])
	solPrimOut[3:,0] = np.array(massFracR).astype(np.float64)[massFracSlice]

solInlet  = solutionPhys(gas, solPrimIn, 1)
solInlet.updateState(fromCons=False)
solOutlet = solutionPhys(gas, solPrimOut, 1) 
solOutlet.updateState(fromCons=False)

# set some variables for ease of use
pressIn = solInlet.solPrim[0,0]
velIn 	= solInlet.solPrim[1,0]
tempIn  = solInlet.solPrim[2,0]
rhoIn 	= solInlet.solCons[0,0] 
CpMixIn = solInlet.CpMix[0]

pressOut = solOutlet.solPrim[0,0]
velOut 	 = solOutlet.solPrim[1,0]
tempOut  = solOutlet.solPrim[2,0]
rhoOut 	 = solOutlet.solCons[0,0]
CpMixOut = solOutlet.CpMix[0]

# calculate sound speed
cIn  = gas.calcSoundSpeed(solInlet.solPrim[2,:], RMix=solInlet.RMix, massFracs=solInlet.solPrim[3:,:], CpMix=solInlet.CpMix)[0]
cOut = gas.calcSoundSpeed(solOutlet.solPrim[2,:], RMix=solOutlet.RMix, massFracs=solOutlet.solPrim[3:,:], CpMix=solOutlet.CpMix)[0]

# reference quantities
pressUp   = pressIn + velIn * rhoIn * cIn 
tempUp    = tempIn + (pressUp - pressIn) / (rhoIn * CpMixIn)
pressBack = pressOut - velOut * rhoOut * cOut

# print results
# TODO: nicer string formatting
print("##### INLET #####")
print("Rho: "+str(rhoIn))
print("Sound speed: "+str(cIn))
print("Cp: "+str(CpMixIn))
print("Rho*C: "+str(rhoIn*cIn))
print("Rho*Cp: "+str(rhoIn*CpMixIn))
print("Upstream pressure: "+str(pressUp))
print("Upstream temp: "+str(tempUp))

print("\n")

print("##### OUTLET #####")
print("Rho: "+str(rhoOut))
print("Sound speed: "+str(cOut))
print("Cp: "+str(CpMixOut))
print("Rho*C: "+str(rhoOut*cOut))
print("Rho*Cp: "+str(rhoOut*CpMixOut))
print("Downstream pressure: "+str(pressBack))

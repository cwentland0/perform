import sys
sys.path.append("..")
import numpy as np
import stateFuncs 
from classDefs import gasProps
from inputFuncs import readInputFile
from math import sqrt
import pdb

##### BEGIN USER INPUT #####

gasFile = "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/standingFlame/Inputs/global1.chem"

fromICFile = False  # If True, load primitive state from file. If False, specify left and right primitive state

icFile = "" 	# primitive state file

# left and right states, if fomrICFile = False
pressL 		= 999775.96996096
velL 		= 11.07569418
tempL 		= 300.03214239
massFracL 	= [1.0, 0.0]
pressR 		= 999743.38742495
velR 		= 16.53076408
tempR 		= 2491.04091458
massFracR 	= [0.0, 1.0]

##### END USER INPUT #####

# load gas file
gas = gasProps(gasFile)

# set up states
if fromICFile:
	solPrim 	= np.load(icFile)
	solPrimIn 	= solPrim[[0],:]
	solPrimOut 	= solPrim[[-1],:]

else:
	solPrimIn 	= np.zeros((1,3+len(massFracL)-1), dtype = np.float64)
	solPrimOut 	= np.zeros((1,3+len(massFracL)-1), dtype = np.float64)
	solPrimIn[0,:3] 	= np.array([pressL, velL, tempL])
	solPrimIn[0,3:] 	= np.array(massFracL).astype(np.float64)[:-1]
	solPrimOut[0,:3] 	= np.array([pressR, velR, tempR])
	solPrimOut[0,3:] 	= np.array(massFracR).astype(np.float64)[:-1]

# calculate conservative state
solConsIn,  RMixIn,  enthRefMixIn,  CpMixIn  = stateFuncs.calcStateFromPrim(solPrimIn, gas)
solConsOut, RMixOut, enthRefMixOut, CpMixOut = stateFuncs.calcStateFromPrim(solPrimOut, gas)

# set some variables for ease of use
pressIn = solPrimIn[0,0]
velIn 	= solPrimIn[0,1]
tempIn  = solPrimIn[0,2]
rhoIn 	= solConsIn[0,0] 

pressOut = solPrimOut[0,0]
velOut 	 = solPrimOut[0,1]
tempOut  = solPrimOut[0,2]
rhoOut 	 = solConsOut[0,0]

# calculate sound speed
gammaIn = stateFuncs.calcGammaMixture(RMixIn, CpMixIn)[0]
cIn 	= np.sqrt(gammaIn * RMixIn[0] * tempIn)
gammaOut = stateFuncs.calcGammaMixture(RMixOut, CpMixOut)[0]
cOut 	= np.sqrt(gammaOut * RMixOut[0] * tempOut)

# reference quantities
pressUp 	= pressIn + velIn * rhoIn * cIn 
tempUp 		= tempIn + (pressUp - pressIn) / (rhoIn * CpMixIn[0])
pressBack 	= pressOut - velOut * rhoOut * cOut

# print results
print("##### INLET #####")
print("Rho: "+str(rhoIn))
print("Sound speed: "+str(cIn))
print("Cp: "+str(CpMixIn[0]))
print("Rho*C: "+str(rhoIn*cIn))
print("Rho*Cp: "+str(rhoIn*CpMixIn[0]))
print("Upstream pressure: "+str(pressUp))
print("Upstream temp: "+str(tempUp))

print("\n")

print("##### OUTLET #####")
print("Rho: "+str(rhoOut))
print("Sound speed: "+str(cOut))
print("Cp: "+str(CpMixOut[0]))
print("Rho*C: "+str(rhoOut*cOut))
print("Rho*Cp: "+str(rhoOut*CpMixOut[0]))
print("Downstream pressure: "+str(pressBack))

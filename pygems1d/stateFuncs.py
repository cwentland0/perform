from constants import realType, hugeNum, RUniv
import numpy as np
import pdb


# compute primitive variables from conservative variables and mixture thermo properties
def calcStateFromCons(solCons, gas):

	# pressure, velocity, temperature, mass fraction
	solPrim = np.zeros(solCons.shape, dtype=realType)

	solPrim[3:,:] 	= solCons[3:,:] / solCons[[0],:] 
	massFracs = gas.getMassFracArray(solPrim=solPrim)

	# update thermo properties
	RMix 			= gas.calcMixGasConstant(massFracs)
	enthRefMix 		= gas.calcMixEnthRef(massFracs)
	CpMix 			= gas.calcMixCp(massFracs)

	# update primitive state
	solPrim[1,:] = solCons[1,:] / solCons[0,:]
	solPrim[2,:] = (solCons[2,:] / solCons[0,:] - np.square(solPrim[1,:])/2.0 - enthRefMix + CpMix * gas.tempRef) / (CpMix - RMix) 
	solPrim[0,:] = solCons[0,:] * RMix * solPrim[2,:]
	
	return solPrim, RMix, enthRefMix, CpMix

# compute conservative variables from primitive variables and mixture thermo properties
def calcStateFromPrim(solPrim, gas):

	# density, momentum, energy, density-weighted mass fraction
	solCons = np.zeros(solPrim.shape, dtype=realType)

	massFracs = gas.getMassFracArray(solPrim=solPrim)

	# update thermo properties
	RMix 			= gas.calcMixGasConstant(massFracs)
	enthRefMix 		= gas.calcMixEnthRef(massFracs)
	CpMix 			= gas.calcMixCp(massFracs)

	# update conservative variables
	solCons[0,:] = solPrim[0,:] / (RMix * solPrim[2,:]) 
	solCons[1,:] = solCons[0,:] * solPrim[1,:]				
	solCons[2,:] = solCons[0,:] * ( enthRefMix + CpMix * (solPrim[2,:] - gas.tempRef) + np.power(solPrim[1,:],2.0) / 2.0 ) - solPrim[0,:]
	solCons[3:,:] = solCons[[0],:]*solPrim[3:,:]

	return solCons, RMix, enthRefMix, CpMix

# Adjust pressure and temperature iterative to agree with a fixed density and stagnation enthalpy
# Used to compute a physically-meaningful Roe average state from the Roe average enthalpy and density 
def calcStateFromRhoH0(solPrim, densFixed, stagEnthFixed, gas):

	densFixed 		= np.squeeze(densFixed)
	stagEnthFixed 	= np.squeeze(stagEnthFixed)

	dPress 		= hugeNum * np.ones(solPrim.shape[1], dtype=np.float64)
	dTemp 		= hugeNum * np.ones(solPrim.shape[1], dtype=np.float64)

	pressCurr 		= solPrim[0,:]

	iterCount = 0
	onesVec = np.ones(solPrim.shape[1], dtype=realType)
	while ( (np.any( np.absolute(dPress / solPrim[0,:]) > 0.01 ) or np.any( np.absolute(dTemp / solPrim[2,:]) > 0.01)) and (iterCount < 20)):

		# compute density and stagnation enthalpy from current state
		densCurr 		= gas.calcDensity(solPrim)
		stagEnthCurr 	= gas.calcStagnationEnthalpy(solPrim)

		# compute difference between current and fixed density/stagnation enthalpy
		dDens 		= densFixed - densCurr 
		dStagEnth 	= stagEnthFixed - stagEnthCurr

		# compute derivatives of density and stagnation enthalpy with respect to pressure and temperature
		DDensDPress, DDensDTemp = gas.calcDensityDerivatives(densCurr, wrtPress=True, pressure=solPrim[0,:], wrtTemp=True, temperature=solPrim[2,:])
		DStagEnthDPress, DStagEnthDTemp = gas.calcStagEnthalpyDerivatives(wrtPress=True, wrtTemp=True, massFracs=solPrim[3:,:])

		# compute change in temperature and pressure 
		dFactor = 1.0 / (DDensDPress * DStagEnthDTemp - DDensDTemp * DStagEnthDPress)
		dPress 	= dFactor * (dDens * DStagEnthDTemp - dStagEnth * DDensDTemp)
		dTemp 	= dFactor * (-dDens * DStagEnthDPress + dStagEnth * DDensDPress)

		# threshold change in temperature and pressure 
		dPress  = np.copysign(onesVec, dPress) * np.minimum(np.absolute(dPress), solPrim[0,:] * 0.1)
		dTemp 	= np.copysign(onesVec, dTemp) * np.minimum(np.absolute(dTemp), solPrim[2,:] * 0.1)

		# update temperature and pressure
		solPrim[0,:] += dPress
		solPrim[2,:] += dTemp

		iterCount += 1

	return solPrim

from pygems1d.constants import realType, RUniv
from pygems1d.higherOrderFuncs import calcCellGradients
from pygems1d.miscFuncs import writeToFile

import numpy as np 
import os
import pdb	


def calcRHS(solDomain, solver):
	"""
	Compute RHS function
	"""

	solInt = solDomain.solInt 
	solIn  = solDomain.solIn 
	solOut = solDomain.solOut

	# compute ghost cell state (if adjacent cell is sampled)
	# TODO: update this after higher-order contribution?
	# TODO: need to adapt solPrim and solCons pass to calcBoundaryState() depending on space scheme
	# TODO: will have to assign more than just one ghost cell for higher-order schemes
	if (solDomain.directSampIdxs[0] == 0):
		solDomain.solIn.calcBoundaryState(solver, solPrim=solInt.solPrim[:,:2], solCons=solInt.solCons[:,:2])
	if (solDomain.directSampIdxs[-1] == (solver.mesh.numCells-1)):
		solDomain.solOut.calcBoundaryState(solver, solPrim=solInt.solPrim[:,-2:], solCons=solInt.solCons[:,-2:])

	solDomain.fillSolFull() # fill solPrimFull and solConsFull

	# first-order approx at faces
	solL = solDomain.solL
	solR = solDomain.solR
	solL.solPrim = solDomain.solPrimFull[:, solDomain.fluxSampLIdxs]
	solL.solCons = solDomain.solConsFull[:, solDomain.fluxSampLIdxs]
	solR.solPrim = solDomain.solPrimFull[:, solDomain.fluxSampRIdxs]
	solR.solCons = solDomain.solConsFull[:, solDomain.fluxSampRIdxs]

	# add higher-order contribution
	if (solver.spaceOrder > 1):
		solPrimGrad = calcCellGradients(solDomain, solver)
		solL.solPrim[:,solDomain.fluxLExtract] += (solver.mesh.dx / 2.0) * solPrimGrad[:, solDomain.gradLExtract]
		solR.solPrim[:,solDomain.fluxRExtract] -= (solver.mesh.dx / 2.0) * solPrimGrad[:, solDomain.gradRExtract]
		solL.calcStateFromPrim(calcR=True, calcEnthRef=True, calcCp=True)
		solR.calcStateFromPrim(calcR=True, calcEnthRef=True, calcCp=True)

	# compute fluxes
	flux = calcInvFlux(solDomain, solver)
	if (solver.viscScheme > 0):
		viscFlux = calcViscFlux(solDomain, solver)
		flux -= viscFlux	

	# compute RHS
	solDomain.solInt.RHS[:, solDomain.directSampIdxs] = flux[:, solDomain.fluxRHSIdxs] - flux[:, solDomain.fluxRHSIdxs+1]
	solInt.RHS[:, solDomain.directSampIdxs] /= solver.mesh.dx

	# compute source term
	if solver.sourceOn:
		calcSource(solDomain, solver)
		solInt.RHS[3:, solDomain.directSampIdxs] += solInt.source[:, solDomain.directSampIdxs]


def calcInvFlux(solDomain, solver):
	"""
	Compute inviscid fluxes
	"""

	# TODO: generalize to other flux schemes, expand beyond Roe flux
	# TODO: entropy fix

	solPrimL = solDomain.solL.solPrim
	solConsL = solDomain.solL.solCons
	solPrimR = solDomain.solR.solPrim
	solConsR = solDomain.solR.solCons

	# inviscid flux vector
	FL = np.zeros(solPrimL.shape, dtype=realType)
	FR = np.zeros(solPrimR.shape, dtype=realType)     

	# compute sqrhol, sqrhor, fac, and fac1
	sqrhol = np.sqrt(solConsL[0,:])
	sqrhor = np.sqrt(solConsR[0,:])
	fac = sqrhol / (sqrhol + sqrhor)
	fac1 = 1.0 - fac

	# Roe average stagnation enthalpy and density
	solDomain.solL.h0 = solDomain.gasModel.calcStagnationEnthalpy(solPrimL)
	solDomain.solR.h0 = solDomain.gasModel.calcStagnationEnthalpy(solPrimR)

	solAve = solDomain.solAve
	solAve.h0 = fac * solDomain.solL.h0 + fac1 * solDomain.solR.h0 
	solAve.solCons[0,:] = sqrhol * sqrhor

	# compute Roe average primitive state, adjust iteratively to conform to Roe average density and enthalpy
	solAve.solPrim = fac[None,:] * solPrimL + fac1[None,:] * solPrimR
	solAve.calcStateFromRhoH0()

	# compute Roe average state at faces, associated fluid properties
	solAve.calcStateFromPrim(calcR=True, calcEnthRef=True, calcCp=True)
	solAve.gammaMix = solDomain.gasModel.calcMixGamma(solAve.RMix, solAve.CpMix)
	solAve.c = solDomain.gasModel.calcSoundSpeed(solAve.solPrim[2,:], RMix=solAve.RMix, gammaMix=solAve.gammaMix,
											 massFracs=solAve.solPrim[3:,:], CpMix=solAve.CpMix)

	# compute inviscid flux vectors of left and right state
	FL[0,:] = solConsL[1,:]
	FL[1,:] = solConsL[1,:] * solPrimL[1,:] + solPrimL[0,:]
	FL[2,:] = solConsL[0,:] * solDomain.solL.h0 * solPrimL[1,:]
	FL[3:,:] = solConsL[3:,:] * solPrimL[[1],:]
	FR[0,:] = solConsR[1,:]
	FR[1,:] = solConsR[1,:] * solPrimR[1,:] + solPrimR[0,:]
	FR[2,:] = solConsR[0,:] * solDomain.solR.h0 * solPrimR[1,:]
	FR[3:,:] = solConsR[3:,:] * solPrimR[[1],:]

	# maximum wave speed for adapting dtau, if needed
	# TODO: need to adaptively size this for hyper-reduction
	if (solDomain.timeIntegrator.adaptDTau):
		srf = np.maximum(solAve.solPrim[1,:] + solAve.c, solAve.solPrim[1,:] - solAve.c)
		solDomain.solInt.srf = np.maximum(srf[:-1], srf[1:])

	# dissipation term
	dQp = solPrimL - solPrimR
	solDomain.RoeDiss = calcRoeDissipation(solAve)
	dissTerm = 0.5 * (solDomain.RoeDiss * np.expand_dims(dQp, 0)).sum(-2)

	# complete Roe flux
	F = 0.5 * (FL + FR) + dissTerm 

	return F


def calcRoeDissipation(solAve):
	"""
	Compute dissipation term of Roe flux
	"""

	# allocate
	dissMat = np.zeros((solAve.gasModel.numEqs, solAve.gasModel.numEqs, solAve.numCells), dtype=realType)        
	
	# primitive variables for clarity
	rho       = solAve.solCons[0,:]
	press     = solAve.solPrim[0,:]
	vel       = solAve.solPrim[1,:]
	temp      = solAve.solPrim[2,:]
	massFracs = solAve.solPrim[3:,:]

	# derivatives of density and enthalpy
	dRho_dP, dRho_dT, dRho_dY = solAve.gasModel.calcDensityDerivatives(solAve.solCons[0,:],
								wrtPress=True, pressure=solAve.solPrim[0,:],
								wrtTemp=True, temperature=solAve.solPrim[2,:],
								wrtSpec=True, massFracs=solAve.solPrim[3:,:])

	dH_dP, dH_dT, dH_dY = solAve.gasModel.calcStagEnthalpyDerivatives(wrtPress=True,
							wrtTemp=True, massFracs=solAve.solPrim[3:,:], 
							wrtSpec=True, temperature=solAve.solPrim[2,:])

	# save for Jacobian calculations
	solAve.dRho_dP = dRho_dP; solAve.dRho_dT = dRho_dT; solAve.dRho_dY = dRho_dY
	solAve.dH_dP = dH_dP; solAve.dH_dT = dH_dT; solAve.dH_dY = dH_dY
	
	# gamma terms for energy equation
	Gp = rho * dH_dP + dRho_dP * solAve.h0 - 1.0
	GT = rho * dH_dT + dRho_dT * solAve.h0
	GY = rho * dH_dY + dRho_dY * solAve.h0

	# characteristic speeds
	lambda1 = vel + solAve.c
	lambda2 = vel - solAve.c
	lam1 = np.absolute(lambda1)
	lam2 = np.absolute(lambda2)

	R_roe = (lam2 - lam1) / (lambda2 - lambda1)
	alpha = solAve.c * (lam1 + lam2) / (lambda1 - lambda2)
	beta = np.power(solAve.c, 2.0) * (lam1 - lam2) / (lambda1 - lambda2)
	phi = solAve.c * (lam1 + lam2) / (lambda1 - lambda2)

	eta = (1.0 - rho * dH_dP) / dH_dT
	psi = eta * dRho_dT + rho * dRho_dP

	u_abs = np.absolute(vel)

	beta_star = beta * psi
	beta_e = beta * (rho * Gp + GT * eta)
	phi_star = dRho_dP * phi + dRho_dT * eta * (phi - u_abs) / rho
	phi_e = Gp * phi + GT * eta * (phi - u_abs) / rho
	m = rho * alpha
	e = rho * vel * alpha

	dissMat[0,0,:] = phi_star
	dissMat[0,1,:] = beta_star
	dissMat[0,2,:] = u_abs * dRho_dT

	dissMat[0,3:,:] = u_abs * dRho_dY
	dissMat[1,0,:] = vel * phi_star + R_roe
	dissMat[1,1,:] = vel * beta_star + m
	dissMat[1,2,:] = vel * u_abs * dRho_dT
	dissMat[1,3:,:] = vel * u_abs * dRho_dY

	dissMat[2,0,:] = phi_e + R_roe * vel
	dissMat[2,1,:] = beta_e + e
	dissMat[2,2,:] = GT * u_abs
	dissMat[2,3:,:] = GY * u_abs

	for yIdx_out in range(3, solAve.gasModel.numEqs):
		dissMat[yIdx_out,0,:] = massFracs[yIdx_out-3,:] * phi_star
		dissMat[yIdx_out,1,:] = massFracs[yIdx_out-3,:] * beta_star
		dissMat[yIdx_out,2,:] = massFracs[yIdx_out-3,:] * u_abs * dRho_dT

		for yIdx_in in range(3, solAve.gasModel.numEqs):
			# TODO: might want to check this again against GEMS, something weird going on
			if (yIdx_out == yIdx_in):
				dissMat[yIdx_out,yIdx_in,:] = u_abs * (rho + massFracs[yIdx_out-3,:] * dRho_dY[yIdx_in-3,:])
			else:
				dissMat[yIdx_out,yIdx_in,:] = u_abs * massFracs[yIdx_out-3,:] * dRho_dY[yIdx_in-3,:]

	return dissMat

 
def calcViscFlux(solDomain, solver):
	"""
	Compute viscous fluxes
	"""

	gas    = solDomain.gasModel 
	mesh   = solver.mesh
	solAve = solDomain.solAve

	# compute 2nd-order state gradients at faces
	# TODO: generalize to higher orders of accuracy
	solPrimGrad = np.zeros((gas.numEqs+1, solDomain.numFluxFaces), dtype=realType)
	solPrimGrad[:-1,:] = (solDomain.solPrimFull[:,solDomain.fluxSampRIdxs] - solDomain.solPrimFull[:,solDomain.fluxSampLIdxs]) / mesh.dx

	# get gradient of last species for diffusion velocity term
	# TODO: maybe a sneakier way to do this?
	massFracs   = gas.calcAllMassFracs(solDomain.solPrimFull[3:,:])
	solPrimGrad[-1,:] = (massFracs[-1,solDomain.fluxSampRIdxs] - massFracs[-1,solDomain.fluxSampLIdxs]) / mesh.dx

	# thermo and transport props
	moleFracs   = gas.calcAllMoleFracs(solAve.solPrim[3:,:])
	specDynVisc = gas.calcSpeciesDynamicVisc(solAve.solPrim[2,:])
	thermCond   = gas.calcMixThermalCond(specDynVisc=specDynVisc, moleFracs=moleFracs)
	muMix       = gas.calcMixDynamicVisc(specDynVisc=specDynVisc, moleFracs=moleFracs)
	massDiff    = gas.calcSpeciesMassDiffCoeff(solAve.solCons[0,:], specDynVisc=specDynVisc)
	hi          = gas.calcSpeciesEnthalpies(solAve.solPrim[2,:])

	tau = 4.0/3.0 * muMix * solPrimGrad[1,:]							# stress "tensor"
	diffVel = solAve.solCons[[0],:] * massDiff * solPrimGrad[3:,:]		# diffusion velocity
	corrVel = np.sum(diffVel, axis=0) 									# correction velocity

	# viscous flux
	Fv = np.zeros((gas.numEqs, solDomain.numFluxFaces), dtype=realType)
	Fv[1,:]  += tau
	Fv[2,:]  += solAve.solPrim[1,:] * tau + thermCond * solPrimGrad[2,:] + np.sum(diffVel * hi, axis = 0)
	Fv[3:,:] += diffVel[gas.massFracSlice] - solAve.solPrim[3:,:] * corrVel[None,:]

	return Fv

 
def calcSource(solDomain, solver):
	"""
	Compute chemical source term
	"""

	# TODO: expand to multiple global reactions

	gas = solDomain.gasModel

	temp 	  = solDomain.solInt.solPrim[2,solDomain.directSampIdxs]
	massFracs = solDomain.solInt.solPrim[3:,solDomain.directSampIdxs]
	rho 	  = solDomain.solInt.solCons[[0],solDomain.directSampIdxs]

	# NOTE: actEnergy here is -Ea/R
	# TODO: account for temperature exponential
	wf = gas.preExpFact * np.exp(gas.actEnergy / temp)
	
	rhoY = massFracs * rho

	# specIdxs = np.argwhere(gas.nuArr != 0.0)
	# wf = wf * np.power((rhoY[specIdx,:] / gas.molWeights[specIdx]), gas.nuArr[specIdx])

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = wf * np.power((rhoY[specIdx,:] / gas.molWeights[specIdx]), gas.nuArr[specIdx])
			wf[massFracs[specIdx, :] < 0.0] = 0.0

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = np.minimum(wf, rhoY[specIdx] / solver.dt)

	# TODO: could vectorize this I think
	for specIdx in range(gas.numSpecies):
		solDomain.solInt.source[specIdx,solDomain.directSampIdxs] = -gas.molWeightNu[specIdx] * wf


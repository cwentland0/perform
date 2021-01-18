
from pygems1d.constants import realType, RUniv
from pygems1d.higherOrderFuncs import calcCellGradients
from pygems1d.miscFuncs import writeToFile

import numpy as np 
import os
import pdb	

# TODO: check for repeated calculations, just make a separate variable
# TODO: check references to muRef, might be some broadcast issues
# TODO: get rid of as many concatenates as possible

# compute RHS function
def calcRHS(solDomain, solver):

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
	# TODO: make this work with gappy POD
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

# compute inviscid fluxes
# TODO: expand beyond Roe flux
# TODO: better naming conventions
# TODO: entropy fix
def calcInvFlux(solDomain, solver):

	solPrimL = solDomain.solL.solPrim
	solConsL = solDomain.solL.solCons
	solPrimR = solDomain.solR.solPrim
	solConsR = solDomain.solR.solCons

	# inviscid flux vector
	EL = np.zeros(solPrimL.shape, dtype=realType)
	ER = np.zeros(solPrimR.shape, dtype=realType)     

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
	EL[0,:] = solConsL[1,:]
	EL[1,:] = solConsL[1,:] * solPrimL[1,:] + solPrimL[0,:]
	EL[2,:] = solConsL[0,:] * solDomain.solL.h0 * solPrimL[1,:]
	EL[3:,:] = solConsL[3:,:] * solPrimL[[1],:]
	ER[0,:] = solConsR[1,:]
	ER[1,:] = solConsR[1,:] * solPrimR[1,:] + solPrimR[0,:]
	ER[2,:] = solConsR[0,:] * solDomain.solR.h0 * solPrimR[1,:]
	ER[3:,:] = solConsR[3:,:] * solPrimR[[1],:]

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
	flux = 0.5 * (EL + ER) + dissTerm 

	return flux

# compute dissipation term of Roe flux
# inputs are all from Roe average state
# TODO: a lot of these quantities need to be generalized for different gas models
def calcRoeDissipation(solAve):

	# allocate
	dissMat = np.zeros((solAve.gasModel.numEqs, solAve.gasModel.numEqs, solAve.numCells), dtype=realType)        
	
	# primitive variables for clarity
	rho       = solAve.solCons[0,:]
	press     = solAve.solPrim[0,:]
	vel       = solAve.solPrim[1,:]
	temp      = solAve.solPrim[2,:]
	massFracs = solAve.solPrim[3:,:]

	# derivatives of density and enthalpy
	rhop, rhoT, rhoY = solAve.gasModel.calcDensityDerivatives(solAve.solCons[0,:],
							wrtPress=True, pressure=solAve.solPrim[0,:],
							wrtTemp=True, temperature=solAve.solPrim[2,:],
							wrtSpec=True, massFracs=solAve.solPrim[3:,:])

	hp, hT, hY = solAve.gasModel.calcStagEnthalpyDerivatives(wrtPress=True,
					wrtTemp=True, massFracs=solAve.solPrim[3:,:], 
					wrtSpec=True, temperature=solAve.solPrim[2,:])

	
	# gamma terms for energy equation
	Gp = rho * hp + rhop * solAve.h0 - 1.0
	GT = rho *hT + rhoT * solAve.h0
	GY = rho * hY + rhoY * solAve.h0

	# characteristic speeds
	lambda1 = vel + solAve.c
	lambda2 = vel - solAve.c
	lam1 = np.absolute(lambda1)
	lam2 = np.absolute(lambda2)

	R_roe = (lam2 - lam1) / (lambda2 - lambda1)
	alpha = solAve.c * (lam1 + lam2) / (lambda1 - lambda2)
	beta = np.power(solAve.c, 2.0) * (lam1 - lam2) / (lambda1 - lambda2)
	phi = solAve.c * (lam1 + lam2) / (lambda1 - lambda2)

	eta = (1.0 - rho * hp) / hT
	psi = eta * rhoT + rho * rhop

	u_abs = np.absolute(vel)

	beta_star = beta * psi
	beta_e = beta * (rho * Gp + GT * eta)
	phi_star = rhop * phi + rhoT * eta * (phi - u_abs) / rho
	phi_e = Gp * phi + GT * eta * (phi - u_abs) / rho
	m = rho * alpha
	e = rho * vel * alpha

	dissMat[0,0,:] = phi_star
	dissMat[0,1,:] = beta_star
	dissMat[0,2,:] = u_abs * rhoT

	dissMat[0,3:,:] = u_abs * rhoY
	dissMat[1,0,:] = vel * phi_star + R_roe
	dissMat[1,1,:] = vel * beta_star + m
	dissMat[1,2,:] = vel * u_abs * rhoT
	dissMat[1,3:,:] = vel * u_abs * rhoY

	dissMat[2,0,:] = phi_e + R_roe * vel
	dissMat[2,1,:] = beta_e + e
	dissMat[2,2,:] = GT * u_abs
	dissMat[2,3:,:] = GY * u_abs

	for yIdx_out in range(3, solAve.gasModel.numEqs):
		dissMat[yIdx_out,0,:] = massFracs[yIdx_out-3,:] * phi_star
		dissMat[yIdx_out,1,:] = massFracs[yIdx_out-3,:] * beta_star
		dissMat[yIdx_out,2,:] = massFracs[yIdx_out-3,:] * u_abs * rhoT

		for yIdx_in in range(3, solAve.gasModel.numEqs):
			# TODO: might want to check this again against GEMS, something weird going on
			if (yIdx_out == yIdx_in):
				dissMat[yIdx_out,yIdx_in,:] = u_abs * (rho + massFracs[yIdx_out-3,:] * rhoY[yIdx_in-3,:])
			else:
				dissMat[yIdx_out,yIdx_in,:] = u_abs * massFracs[yIdx_out-3,:] * rhoY[yIdx_in-3,:]

	return dissMat

# compute viscous fluxes
def calcViscFlux(solDomain, solver):

	gas 	= solDomain.gasModel 
	mesh 	= solver.mesh
	solInt  = solDomain.solInt

	# compute 2nd-order state gradients at faces
	# TODO: generalize to higher orders of accuracy
	solPrimGrad = np.zeros((gas.numEqs, solDomain.numFluxFaces), dtype=realType)
	solPrimGrad = (solDomain.solPrimFull[:,solDomain.fluxSampRIdxs] - solDomain.solPrimFull[:,solDomain.fluxSampLIdxs]) / mesh.dx

	# TODO: gasModel refs
	Ck  = gas.muRef[gas.massFracSlice] * CpAve / gas.Pr[gas.massFracSlice] 				# thermal conductivity
	tau = 4.0/3.0 * gas.muRef[gas.massFracSlice] * solPrimGrad[1,:] 					# stress "tensor"

	Cd        = gas.muRef[gas.massFracSlice] / gas.Sc[gas.massFracSlice] / solConsAve[0,:]		# mass diffusivity
	diff_rhoY = solConsAve[0,:] * Cd * np.squeeze(solPrimGrad[3:,:])  					# 
	hi        = gas.enthRefDiffs + (solPrimAve[2,:] - gas.tempRef) * gas.CpDiffs 				# species enthalpies, TODO: replace with gas model function

	Fv = np.zeros((gas.numEqs, solDomain.numFluxFaces), dtype=realType)
	Fv[1,:] = Fv[1,:] + tau 
	Fv[2,:] = Fv[2,:] + solPrimAve[1,:] * tau + Ck * solPrimGrad[2,:]
	if (gas.numSpecies > 1):
		Fv[2,:] = Fv[2,:] + np.sum(diff_rhoY * hY, axis = 0)
		Fv[3:,:] = Fv[3:,:] + diff_rhoY 
	else:
		Fv[2,:] = Fv[2,:] + diff_rhoY * hY
		Fv[3,:] = Fv[3,:] + diff_rhoY

	return Fv

# compute source term
# TODO: bring in rho*Yi so it doesn't keep getting calculated
def calcSource(solDomain, solver):

	gas = solDomain.gasModel

	temp 	  = solDomain.solInt.solPrim[2,solDomain.directSampIdxs]
	massFracs = solDomain.solInt.solPrim[3:,solDomain.directSampIdxs]
	rho 	  = solDomain.solInt.solCons[[0],solDomain.directSampIdxs]

	wf = gas.preExpFact * np.exp(gas.actEnergy / temp)
	
	rhoY = massFracs * rho

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


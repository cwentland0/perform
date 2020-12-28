import numpy as np 
import constants
import os
import struct
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries
import stateFuncs
from boundaryFuncs import calcBoundaries
import pdb	

# TODO: check for repeated calculations, just make a separate variable
# TODO: check references to muRef, might be some broadcast issues

# compute RHS function
# @profile
def calcRHS(sol: solutionPhys, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps):

	# compute ghost cell state or boundary fluxes
	# TODO: update this after higher-order contribution?
	calcBoundaries(sol, bounds, params, gas)

	# state at left and right of cell face
	solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
	solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
	solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
	solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)       

	# add higher-order contribution
	if (params.spaceOrder > 1):
		solPrimGrad = calcCellGradients(sol, params, bounds, geom, gas)
		solPrimL[1:,:] 	+= (geom.dx / 2.0) * solPrimGrad 
		solPrimR[:-1,:] -= (geom.dx / 2.0) * solPrimGrad
		solConsL[1:,:], _, _ ,_ = stateFuncs.calcStateFromPrim(solPrimL[1:,:], gas)
		solConsR[:-1,:], _, _ ,_ = stateFuncs.calcStateFromPrim(solPrimR[:-1,:], gas)

	pdb.set_trace()

	# compute fluxes
	flux, solPrimAve, solConsAve, CpAve = calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol, params, gas)
	if (params.viscScheme > 0):
		flux -= calcViscFlux(sol, solPrimAve, solConsAve, CpAve, bounds, params, gas, geom)

	# compute RHS
	sol.RHS = flux[:-1,:] - flux[1:,:]
	sol.RHS[:,:] /= geom.dx	

	# compute source term
	if params.sourceOn:
		calcSource(sol, params, gas)
		sol.RHS[:,3:] += sol.source 

# compute inviscid fluxes
# TODO: expand beyond Roe flux
# TODO: better naming conventions
# TODO: entropy fix
def calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol: solutionPhys, params: parameters, gas: gasProps):

	# TODO: check for non-physical cells
	matShape = solPrimL.shape

	# inviscid flux vector
	EL = np.zeros(matShape, dtype = constants.realType)
	ER = np.zeros(matShape, dtype = constants.realType)     

	# compute sqrhol, sqrhor, fac, and fac1
	sqrhol = np.sqrt(solConsL[:, 0])
	sqrhor = np.sqrt(solConsR[:, 0])
	fac = sqrhol / (sqrhol + sqrhor)
	fac1 = 1.0 - fac

	# Roe average stagnation enthalpy and density
	h0L = stateFuncs.calcStagnationEnthalpy(solPrimL, gas)
	h0R = stateFuncs.calcStagnationEnthalpy(solPrimR, gas) 
	h0Ave = fac * h0L + fac1 * h0R 
	rhoAve = sqrhol * sqrhor

	# compute Roe average primitive state, adjust iteratively to conform to Roe average density and enthalpy
	solPrimAve = fac[:,None] * solPrimL + fac1[:,None] * solPrimR
	solPrimAve = stateFuncs.calcStateFromRhoH0(solPrimAve, rhoAve, h0Ave, gas)

	# compute Roe average state at faces, associated fluid properties
	solConsAve, RAve, enthRefAve, CpAve = stateFuncs.calcStateFromPrim(solPrimAve, gas)
	gammaAve = stateFuncs.calcGammaMixture(RAve, CpAve)
	cAve = np.sqrt(gammaAve * RAve * solPrimAve[:,2])

	# compute inviscid flux vectors of left and right state
	EL[:,0] = solConsL[:,1]
	EL[:,1] = solConsL[:,1] * solPrimL[:,1] + solPrimL[:,0]
	EL[:,2] = solConsL[:,0] * h0L * solPrimL[:,1]
	EL[:,3:] = solConsL[:,3:] * solPrimL[:,[1]]
	ER[:,0] = solConsR[:,1]
	ER[:,1] = solConsR[:,1] * solPrimR[:,1] + solPrimR[:,0]
	ER[:,2] = solConsR[:,0] * h0R * solPrimR[:,1]
	ER[:,3:] = solConsR[:,3:] * solPrimR[:,[1]]

	# maximum wave speed for adapting dtau, if needed
	if (params.adaptDTau):
		srf = np.maximum(solPrimAve[:,1] + cAve, solPrimAve[:,1] - cAve)
		sol.srf = np.maximum(srf[:-1], srf[1:])

	# dissipation term
	dQp = solPrimL - solPrimR
	M_ROE = calcRoeDissipation(solPrimAve, solConsAve[:,0], h0Ave, cAve, CpAve, gas)
	dissTerm = 0.5 * (M_ROE * np.expand_dims(dQp, -2)).sum(-1)

	# complete Roe flux
	flux = 0.5 * (EL + ER) + dissTerm 

	return flux, solPrimAve, solConsAve, CpAve

# compute dissipation term of Roe flux
# inputs are all from Roe average state
# TODO: a lot of these quantities need to be generalized for different gas models
def calcRoeDissipation(solPrim, rho, h0, c, Cp, gas: gasProps):

	# allocate
	dissMat = np.zeros((solPrim.shape[0], gas.numEqs, gas.numEqs), dtype = constants.realType)
	if (solPrim.dtype == constants.complexType):
		dissMat = np.zeros((solPrim.shape[0], gas.numEqs, gas.numEqs), dtype = constants.complexType)        
	
	# primitive variables for clarity
	press = solPrim[:,0]
	vel = solPrim[:,1]
	temp = solPrim[:,2]
	massFracs = solPrim[:,3:]

	rhoY = -np.square(rho) * (constants.RUniv * temp / press * gas.mwInvDiffs)
	hY = gas.enthRefDiffs + (temp - gas.tempRef) * gas.CpDiffs

	rhop = rho / press 					# derivative of density with respect to pressure
	rhoT = -rho / temp 					# derivative of density with respect to temperature
	hT = Cp 							# derivative of enthalpy with respect to temperature
	hp = 0.0 							# derivative of enthalpy with respect to pressure

	# gamma terms for energy equation
	Gp = rho * hp + rhop * h0 - 1.0
	GT = rho *hT + rhoT * h0
	GY = rho * hY + rhoY * h0

	# characteristic speeds
	lambda1 = vel + c
	lambda2 = vel - c
	lam1 = np.absolute(lambda1)
	lam2 = np.absolute(lambda2)

	R_roe = (lam2 - lam1) / (lambda2 - lambda1)
	alpha = c * (lam1 + lam2) / (lambda1 - lambda2)
	beta = np.power(c, 2.0) * (lam1 - lam2) / (lambda1 - lambda2)
	phi = c * (lam1 + lam2) / (lambda1 - lambda2)

	eta = (1.0 - rho * hp) / hT
	psi = eta * rhoT + rho * rhop

	u_abs = np.absolute(vel)

	beta_star = beta * psi
	beta_e = beta * (rho * Gp + GT * eta)
	phi_star = rhop * phi + rhoT * eta * (phi - u_abs) / rho
	phi_e = Gp * phi + GT * eta * (phi - u_abs) / rho
	m = rho * alpha
	e = rho * vel * alpha

	dissMat[:,0,0] = phi_star
	dissMat[:,0,1] = beta_star
	dissMat[:,0,2] = u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[:,0,3:] = u_abs * rhoY
	else:
		dissMat[:,0,3] = u_abs * rhoY
	dissMat[:,1,0] = vel * phi_star + R_roe
	dissMat[:,1,1] = vel * beta_star + m
	dissMat[:,1,2] = vel * u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[:,1,3:] = vel * u_abs * rhoY
	else:
		dissMat[:,1,3] = vel * u_abs * rhoY

	dissMat[:,2,0] = phi_e + R_roe * vel
	dissMat[:,2,1] = beta_e + e
	dissMat[:,2,2] = GT * u_abs

	if (gas.numSpecies > 1):
		dissMat[:,2,3:] = GY * u_abs
	else:
		dissMat[:,2,3] = GY * u_abs

	for yIdx_out in range(3, gas.numEqs):
		dissMat[:, yIdx_out, 0] = massFracs[:, yIdx_out-3] * phi_star
		dissMat[:, yIdx_out, 1] = massFracs[:, yIdx_out-3] * beta_star
		dissMat[:, yIdx_out, 2] = massFracs[:, yIdx_out-3] * u_abs * rhoT

		for yIdx_in in range(3, gas.numEqs):
			# TODO: rhoY is currently calculated incorrectly for multiple species, only works for two species 
			# 		In a working model, rhoY should be rhoY[:, yIdxs_in - 3]
			if (yIdx_out == yIdx_in):
				dissMat[:, yIdx_out, yIdx_in] = u_abs * (rho + massFracs[:, yIdx_out-3] * rhoY)
			else:
				dissMat[:, yIdx_out, yIdx_in] = u_abs * massFracs[:, yIdx_out-3] * rhoY

	return dissMat

# compute viscous fluxes
def calcViscFlux(sol: solutionPhys, solPrimAve, solConsAve, CpAve, bounds: boundaries, params: parameters, gas: gasProps, geom: geometry):

	# compute 2nd-order state gradients at face
	solPrimGrad = np.zeros((geom.numCells+1, gas.numEqs), dtype = constants.realType)
	solPrimGrad[1:-1,:] = (sol.solPrim[1:, :] - sol.solPrim[:-1, :]) / geom.dx
	solPrimGrad[0,:] 	= (sol.solPrim[0, :] - bounds.inlet.sol.solPrim) / geom.dx 
	solPrimGrad[-1,:] 	= (bounds.outlet.sol.solPrim - sol.solPrim[-1,:]) / geom.dx

	Ck = gas.muRef[:-1] * CpAve / gas.Pr[:-1] 									# thermal conductivity
	tau = 4.0/3.0 * gas.muRef[:-1] * solPrimGrad[:,1] 							# stress "tensor"

	Cd = gas.muRef[:-1] / gas.Sc[:-1] / solConsAve[:,0]							# mass diffusivity
	diff_rhoY = solConsAve[:,0] * Cd * np.squeeze(solPrimGrad[:,3:])  			# 
	hY = gas.enthRefDiffs + (solPrimAve[:,2] - gas.tempRef) * gas.CpDiffs 		# species enthalpies, TODO: replace with stateFuncs function

	Fv = np.zeros((geom.numCells+1, gas.numEqs), dtype = constants.realType)
	Fv[:,1] = Fv[:,1] + tau 
	Fv[:,2] = Fv[:,2] + solPrimAve[:,1] * tau + Ck * solPrimGrad[:,2]
	if (gas.numSpecies > 1):
		Fv[:,2] = Fv[:,2] + np.sum(diff_rhoY * hY, axis = 1)
		Fv[:,3:] = Fv[:,3:] + diff_rhoY 
	else:
		Fv[:,2] = Fv[:,2] + diff_rhoY * hY
		Fv[:,3] = Fv[:,3] + diff_rhoY

	return Fv

# compute source term
# TODO: bring in rho*Yi so it doesn't keep getting calculated
def calcSource(sol: solutionPhys, params: parameters, gas: gasProps):

	temp = sol.solPrim[:,2]
	massFracs = sol.solPrim[:,3:]
	if (sol.solPrim.dtype == constants.complexType): #for complex step check
		sol.source = np.zeros(sol.source.shape, dtype = constants.complexType)
	wf = gas.preExpFact * np.exp(gas.actEnergy / temp)
	
	rhoY = massFracs * sol.solCons[:,[0]]

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = wf * np.power((rhoY[:,specIdx] / gas.molWeights[specIdx]), gas.nuArr[specIdx])
			wf[massFracs[:, specIdx] < 0.0] = 0.0

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = np.minimum(wf, rhoY[specIdx] / params.dt)

	# TODO: could vectorize this I think
	for specIdx in range(gas.numSpecies):
		sol.source[:,specIdx] = -gas.molWeightNu[specIdx] * wf

def calcCellGradients(sol: solutionPhys, params: parameters, bounds: boundaries, geom: geometry, gas: gasProps):
	
	# compute gradients via a finite difference stencil
	solPrimGrad = np.zeros(sol.solPrim.shape, dtype=constants.realType)
	if (params.spaceOrder == 2):
		solPrimGrad[1:-1, :] = (0.5 / geom.dx) * (sol.solPrim[2:, :] - sol.solPrim[:-2, :]) 
		solPrimGrad[0, :]    = (0.5 / geom.dx) * (sol.solPrim[1,:] - bounds.inlet.sol.solPrim)
		solPrimGrad[-1,:]    = (0.5 / geom.dx) * (bounds.outlet.sol.solPrim - sol.solPrim[-2,:])
	else:
		raise ValueError("Order "+str(params.spaceOrder)+" gradient calculations not implemented...")

	# compute gradient limiter and limit gradient, if necessary
	if (params.gradLimiter > 0):

		# Barth-Jespersen
		if (params.gradLimiter == 1):
			phi = limiterBarthJespersen(sol, bounds, geom, solPrimGrad)

		elif (params.gradLimiter == 2):
			phi = limiterVenkatakrishnan(sol, bounds, geom, solPrimGrad)

		else:
			raise ValueError("Invalid input for params.limiter: "+str(params.gradLimiter))
	
		solPrimGrad = solPrimGrad * phi	# limit gradient

	return solPrimGrad
	
# find minimum and maximum of cell state and neighbor cell state
def findNeighborMinMax(solInterior, solInlet=None, solOutlet=None):

	# max and min of cell and neighbors
	solMax = solInterior.copy()
	solMin = solInterior.copy()

	# first compare against right neighbor
	solMax[:-1,:]  	= np.maximum(solInterior[:-1,:], solInterior[1:,:])
	solMin[:-1,:]  	= np.minimum(solInterior[:-1,:], solInterior[1:,:])
	if (solOutlet is not None):
		solMax[-1,:]	= np.maximum(solInterior[-1,:], solOutlet[0,:])
		solMin[-1,:]	= np.minimum(solInterior[-1,:], solOutlet[0,:])

	# then compare agains left neighbor
	solMax[1:,:] 	= np.maximum(solMax[1:,:], solInterior[:-1])
	solMin[1:,:] 	= np.minimum(solMin[1:,:], solInterior[:-1])
	if (solInlet is not None):
		solMax[0,:] 	= np.maximum(solMax[0,:], solInlet[0,:])
		solMin[0,:] 	= np.minimum(solMin[0,:], solInlet[0,:])

	return solMin, solMax

# Barth-Jespersen limiter
# ensures that no new minima or maxima are introduced in reconstruction
def limiterBarthJespersen(sol: solutionPhys, bounds: boundaries, geom: geometry, grad):

	solPrim = sol.solPrim

	# get min/max of cell and neighbors
	solPrimMin, solPrimMax = findNeighborMinMax(solPrim, bounds.inlet.sol.solPrim, bounds.outlet.sol.solPrim)

	# unconstrained reconstruction at each face
	delSolPrim 		= grad * (geom.dx / 2) 
	solPrimL 		= solPrim - delSolPrim
	solPrimR 		= solPrim + delSolPrim
	
	# limiter defaults to 1
	phiL = np.ones(solPrim.shape, dtype=constants.realType)
	phiR = np.ones(solPrim.shape, dtype=constants.realType)
	
	# find indices where difference in reconstruction is either positive or negative
	cond1L = ((solPrimL - solPrim) > 0)
	cond1R = ((solPrimR - solPrim) > 0)
	cond2L = ((solPrimL - solPrim) < 0)
	cond2R = ((solPrimR - solPrim) < 0)

	# threshold limiter for left and right faces of each cell
	phiL[cond1L] = np.minimum(phiL[cond1L], (solPrimMax[cond1L] - solPrim[cond1L]) / (solPrimL[cond1L] - solPrim[cond1L]))
	phiR[cond1R] = np.minimum(phiR[cond1R], (solPrimMax[cond1R] - solPrim[cond1R]) / (solPrimR[cond1R] - solPrim[cond1R]))
	phiL[cond2L] = np.minimum(phiL[cond2L], (solPrimMin[cond2L] - solPrim[cond2L]) / (solPrimL[cond2L] - solPrim[cond2L]))
	phiR[cond2R] = np.minimum(phiR[cond2R], (solPrimMin[cond2R] - solPrim[cond2R]) / (solPrimR[cond2R] - solPrim[cond2R]))

	# pdb.set_trace()
	# take minimum limiter from left and right faces of each cell
	phi = np.minimum(phiL, phiR)
	
	return phi

# Venkatakrishnan limiter
# differentiable, but limits in uniform regions
def limiterVenkatakrishnan(sol: solutionPhys, bounds: boundaries, geom: geometry, grad):

	solPrim = sol.solPrim

	# get min/max of cell and neighbors
	solPrimMin, solPrimMax = findNeighborMinMax(solPrim, bounds.inlet.sol.solPrim, bounds.outlet.sol.solPrim)

	# unconstrained reconstruction at each face
	delSolPrim 		= grad * (geom.dx / 2) 
	solPrimL 		= solPrim - delSolPrim
	solPrimR 		= solPrim + delSolPrim
	
	# limiter defaults to 1
	phiL = np.ones(solPrim.shape, dtype=constants.realType)
	phiR = np.ones(solPrim.shape, dtype=constants.realType)
	
	# find indices where difference in reconstruction is either positive or negative
	cond1L = ((solPrimL - solPrim) > 0)
	cond1R = ((solPrimR - solPrim) > 0)
	cond2L = ((solPrimL - solPrim) < 0)
	cond2R = ((solPrimR - solPrim) < 0)
	
	# (y^2 + 2y) / (y^2 + y + 2)
	def venkatakrishnanFunction(maxVals, cellVals, faceVals):
		frac = (maxVals - cellVals) / (faceVals - cellVals)
		fracSq = np.square(frac)
		venkVals = (fracSq + 2.0 * frac) / (fracSq + frac + 2.0)
		return venkVals

	# apply smooth Venkatakrishnan function
	phiL[cond1L] = venkatakrishnanFunction(solPrimMax[cond1L], solPrim[cond1L], solPrimL[cond1L]) 
	phiR[cond1R] = venkatakrishnanFunction(solPrimMax[cond1R], solPrim[cond1R], solPrimR[cond1R]) 
	phiL[cond2L] = venkatakrishnanFunction(solPrimMin[cond2L], solPrim[cond2L], solPrimL[cond2L])
	phiR[cond2R] = venkatakrishnanFunction(solPrimMin[cond2R], solPrim[cond2R], solPrimR[cond2R])

	# take minimum limiter from left and right faces of each cell
	phi = np.minimum(phiL, phiR)
	# pdb.set_trace()

	return phi
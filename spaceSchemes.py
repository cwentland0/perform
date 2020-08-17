import numpy as np 
import constants
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries
from stateFuncs import calcGammaMixture, calcCpMixture, calcGasConstantMixture
from boundaryFuncs import calcBoundaries
import pdb	

# TODO: check for repeated calculations, just make a separate variable
# TODO: check references to muRef, might be some broadcast issues

# compute RHS function
# @profile
def calcRHS(sol: solutionPhys, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps):

	# compute ghost cell state or boundary fluxes
	calcBoundaries(sol, bounds, params, gas)

	# store left and right cell states
	# TODO: this is memory-inefficient, left and right state are not mutations from the state vectors
	if (params.spaceOrder > 1):
		raise ValueError("Higher-order fluxes not implemented yet") 
		
	# if using strong boundary conditions, append ghost cells to state vectors
	if not params.weakBCs:
		solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
		solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
		solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
		solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)
	
	else:
		solPrimL = sol.solPrim[:-1, :]
		solConsL = sol.solCons[:-1, :]
		solPrimR = sol.solPrim[1:, :]
		solConsR = sol.solCons[1:, :]

	# compute fluxes
	flux = calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, gas)
	if (params.viscScheme == 1):
		flux -= calcViscFlux(sol, bounds, params, gas, geom)

	# compute source term
	source = calcSource(sol.solPrim, sol.solCons[:,0], params, gas)

	# compute RHS
	sol.RHS[:,:] 	= 0.0 
	if params.weakBCs:
		sol.RHS[0,:] 	= bounds.inlet.flux
		sol.RHS[-1,:] 	= -bounds.outlet.flux
		sol.RHS[1:,:]   = sol.RHS[1:,:] + flux 
		sol.RHS[:-1,:]  = sol.RHS[:-1,:] - flux
	else:
		sol.RHS = flux[:-1,:] - flux[1:,:]

	sol.RHS[:,:] /= geom.dx	
	sol.RHS  = source + sol.RHS

# compute inviscid fluxes
# TODO: expand beyond Roe flux
# TODO: better naming conventions
# TODO: entropy fix
# @profile
def calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, gas: gasProps):

	# TODO: check for non-physical cells
	matShape = solPrimL.shape

	# allocations
	EL 			= np.zeros(matShape, dtype = constants.floatType)
	ER 			= np.zeros(matShape, dtype = constants.floatType)

	# left flux
	rHL = solConsL[:,[2]] + solPrimL[:,[0]]
	HL = rHL / solConsL[:,[0]]
	EL[:,0] = solConsL[:,1]
	EL[:,1] = solConsL[:,1] * solPrimL[:,1] + solPrimL[:,0]
	EL[:,[2]] = rHL * solPrimL[:,[1]]
	EL[:,3:] = solConsL[:,3:] * solPrimL[:,[1]]

	# right flux
	rHR = solConsR[:,[2]] + solPrimR[:,[0]]
	HR = rHR / solConsR[:,[0]]
	ER[:,0] = solConsR[:,1]
	ER[:,1] = solConsR[:,1] * solPrimR[:,1] + solPrimR[:,0]
	ER[:,[2]] = rHR * solPrimR[:,[1]]
	ER[:,3:] = solConsR[:,3:] * solPrimR[:,[1]]

	rhoi = np.sqrt(solConsR[:,0] * solConsL[:,0])
	di = np.sqrt(solConsR[:,[0]] / solConsL[:,[0]])
	dl = 1.0 / (1.0 + di) 

	solPrimRoe = (solPrimR * di + solPrimL) * dl
	Hi = np.squeeze((di * HR + HL) * dl)

	if (gas.numSpecies > 1):
		massFracsRoe = solPrimRoe[:,3:]
	else:
		massFracsRoe = solPrimRoe[:,3]

	Ri = calcGasConstantMixture(massFracsRoe, gas)
	Cpi = calcCpMixture(massFracsRoe, gas)
	gammai = calcGammaMixture(Ri, Cpi)

	ci = np.sqrt(gammai * Ri * solPrimRoe[:,2])

	dQp = solPrimR - solPrimL
	M_ROE = calcRoeDissipation(solPrimRoe, rhoi, Hi, ci, Ri, Cpi, gas)

	dissTerm = 0.5 * (M_ROE * np.expand_dims(dQp, -2)).sum(-1)

	flux = 0.5 * (EL + ER) - dissTerm 

	return flux

# compute dissipation term of Roe flux
# TODO: better naming conventions
def calcRoeDissipation(solPrim, rho, h0, c, R, Cp, gas: gasProps):

	dissMat = np.zeros((solPrim.shape[0], gas.numEqs, gas.numEqs), dtype = constants.floatType)
	temp = solPrim[:,2]

	# TODO: use the relevant stateFunc
	massFracs = solPrim[:,3:]
	rhoY = -np.square(rho) * (constants.RUniv * temp / solPrim[:,0] * gas.mwDiffs)
	hY = gas.enthRefDiffs + (temp - gas.tempRef) * gas.CpDiffs

	rhop = 1.0 / (R * temp)
	rhoT = -rho / temp
	hT = Cp
	hp = 0.0

	Gp = rho * hp + rhop * h0 - 1.0
	GT = rho *hT + rhoT * h0

	GY = rho * hY + rhoY * h0

	u = solPrim[:,1]
	lambda1 = u + c
	lambda2 = u - c
	lam1 = np.absolute(lambda1)
	lam2 = np.absolute(lambda2)
	R_roe = (lam2 - lam1) / (lambda2 - lambda1)
	alpha = c * (lam1 + lam2) / (lambda1 - lambda2)
	beta = np.power(c, 2.0) * (lam1 - lam2) / (lambda1 - lambda2)
	phi = c * (lam1 + lam2) / (lambda1 - lambda2)

	eta = (1.0 - rho * hp) / hT
	psi = eta * rhoT + rho * rhop

	u_abs = np.absolute(solPrim[:,1])

	beta_star = beta * psi
	beta_e = beta * (rho * Gp + GT * eta)
	phi_star = rhop * phi + rhoT * eta * (phi - u_abs)
	phi_e = Gp * phi + GT * eta * (phi - u_abs)
	m = rho * alpha
	e = rho * u * alpha

	dissMat[:,0,0] = phi_star
	dissMat[:,0,1] = beta_star
	dissMat[:,0,2] = u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[:,0,3:] = u_abs * rhoY
	else:
		dissMat[:,0,3] = u_abs * rhoY
	dissMat[:,1,0] = u *phi_star + R_roe
	dissMat[:,1,1] = u * beta_star + m
	dissMat[:,1,2] = u * u_abs * rhoT

	if (gas.numSpecies > 1):
		dissMat[:,1,3:] = u * u_abs * rhoY
	else:
		dissMat[:,1,3] = u * u_abs * rhoY

	dissMat[:,2,0] = phi_e + R_roe * u
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
def calcViscFlux(sol: solutionPhys, bounds: boundaries, params: parameters, gas: gasProps, geom: geometry):

	# for weak BCs, viscous effects are ignored
	if params.weakBCs:
		solPrim = sol.solPrim
		rho = sol.solCons[:,0]
		cp = sol.CpMix

	else:
		solPrim = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim, bounds.outlet.sol.solPrim), axis = 0)
		rho = np.concatenate((bounds.inlet.sol.solCons[[0],0], sol.solCons[:,0], bounds.outlet.sol.solCons[[0],0]), axis = 0)
		cp = np.concatenate((bounds.inlet.sol.CpMix, sol.CpMix, bounds.outlet.sol.CpMix), axis = 0)
	
	# TODO: gradients should conform with higher-order gradients
	solPrimGrad = np.zeros(solPrim.shape, dtype = constants.floatType)
	solPrimGrad[1:-1, :] = (0.5 / geom.dx) * (solPrim[2:, :] - solPrim[:-2, :]) 
	solPrimGrad[0,:] = (solPrim[1,:] - solPrim[0,:]) / geom.dx
	solPrimGrad[-1,:] = (solPrim[-1,:] - solPrim[-2,:]) / geom.dx 

	Fv = np.zeros(solPrim.shape, dtype = constants.floatType)

	Ck = gas.muRef[:-1] * cp / gas.Pr[:-1]
	tau = 4.0/3.0 * gas.muRef[:-1] * solPrimGrad[:,1]
	Fv[:,1] = Fv[:,1] + tau
	Fv[:,2] = Fv[:,2] + solPrim[:,1] * tau + Ck * solPrimGrad[:,2]

	Cd = gas.muRef[:-1] / gas.Sc[:-1] / rho
	diff_rhoY = rho * Cd * np.squeeze(solPrimGrad[:,3:])
	hY = gas.enthRefDiffs + (solPrim[:,2] - gas.tempRef) * gas.CpDiffs

	if (gas.numSpecies > 1):
		Fv[:,2] = Fv[:,2] + np.sum(diff_rhoY * hY, axis = 1)
		Fv[:,3:] = Fv[:,3:] + diff_rhoY
	else:
		Fv[:,2] = Fv[:,2] + diff_rhoY * hY
		Fv[:,3] = Fv[:,3] + diff_rhoY
   	
	flux = 0.5 * (Fv[:-1,:] + Fv[1:,:])

	return flux

# compute source term
# TODO: bring in rho*Yi so it doesn't keep getting calculated
def calcSource(solPrim, rho, params: parameters, gas: gasProps):

	temp = solPrim[:,2]
	massFracs = solPrim[:,3:]
	sourceTerm = np.zeros(solPrim.shape, dtype = constants.floatType)
	wf = gas.preExpFact * np.exp(gas.actEnergy / temp)
	
	# to avoid recalculating
	rhoY = massFracs * rho[:, np.newaxis]

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = wf * np.power((rhoY[:,specIdx] / gas.molWeights[specIdx]), gas.nuArr[specIdx])
			wf[massFracs[:, specIdx] < 0.0] = 0.0

	for specIdx in range(gas.numSpecies):
		if (gas.nuArr[specIdx] != 0.0):
			wf = np.minimum(wf, rhoY[specIdx] / params.dt)

	# TODO: could vectorize this I think
	for specIdx in range(gas.numSpecies):
		sourceTerm[:,3+specIdx] = -gas.molWeightNu[specIdx] * wf

	# TODO: implicit implementation
	if (params.timeType == "implicit"):
		raise ValueError("Implicit source term needs to be implemented!")
	elif (params.timeType == "explicit"):
		return sourceTerm

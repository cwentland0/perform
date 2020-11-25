import numpy as np 
import constants
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries
from stateFuncs import calcGammaMixture, calcCpMixture, calcGasConstantMixture, calcStateFromPrim
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
	if (params.spaceOrder == 1):
		solPrimL = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim), axis=0)
		solConsL = np.concatenate((bounds.inlet.sol.solCons, sol.solCons), axis=0)
		solPrimR = np.concatenate((sol.solPrim, bounds.outlet.sol.solPrim), axis=0)
		solConsR = np.concatenate((sol.solCons, bounds.outlet.sol.solCons), axis=0)       
		faceVals = None

	elif (params.spaceOrder == 2):
		[solPrimL, solConsL, solPrimR, solConsR, faceVals] = reconstruct_2nd(sol,bounds,geom,gas)  
	else:
		raise ValueError("Higher-order fluxes not implemented yet")

	if (sol.solPrim.dtype == constants.complexType):
		solPrimL = solPrimL.astype(dtype=constants.complexType)
		solPrimR = solPrimR.astype(dtype=constants.complexType)
		solConsL = solConsL.astype(dtype=constants.complexType)
		solConsR = solConsR.astype(dtype=constants.complexType)

	# compute fluxes
	flux = calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol, params, gas)
	if (params.viscScheme == 1):
		flux -= calcViscFlux(sol, bounds, params, gas, geom, faceVals)

	# compute source term
	calcSource(sol, params, gas)

	# compute RHS
	sol.RHS = flux[:-1,:] - flux[1:,:]
	sol.RHS[:,:] /= geom.dx	
	sol.RHS[:,3:]  = sol.source + sol.RHS[:,3:]

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

	if (solPrimL.dtype == constants.complexType):
		EL = np.zeros(matShape, dtype = constants.complexType)        
		ER = np.zeros(matShape, dtype = constants.complexType)        

	# left flux
	rHL = solConsL[:,[2]] + solPrimL[:,[0]]
	HL = rHL / solConsL[:,[0]]								# stagnation enthalpy
	EL[:,0] = solConsL[:,1]
	EL[:,1] = solConsL[:,1] * solPrimL[:,1] + solPrimL[:,0]
	EL[:,[2]] = rHL * solPrimL[:,[1]]
	EL[:,3:] = solConsL[:,3:] * solPrimL[:,[1]]

	# right flux
	rHR = solConsR[:,[2]] + solPrimR[:,[0]]
	HR = rHR / solConsR[:,[0]]								# stagnation enthalpy
	ER[:,0] = solConsR[:,1]
	ER[:,1] = solConsR[:,1] * solPrimR[:,1] + solPrimR[:,0]
	ER[:,[2]] = rHR * solPrimR[:,[1]]
	ER[:,3:] = solConsR[:,3:] * solPrimR[:,[1]]


	rhoi = np.sqrt(solConsR[:,0] * solConsL[:,0]) 		# Roe average density
	di = np.sqrt(solConsR[:,[0]] / solConsL[:,[0]]) 	# sqrt density quotients
	dl = 1.0 / (1.0 + di) 								#

	solPrimRoe = (solPrimR * di + solPrimL) * dl 		# Roe average primitive state
	Hi = np.squeeze((HR * di + HL) * dl) 				# Roe average stagnation enthalpy

	if (gas.numSpecies > 1):
		massFracsRoe = solPrimRoe[:,3:]
	else:
		massFracsRoe = solPrimRoe[:,3]

	# Roe average state mixture gas properties
	Ri = calcGasConstantMixture(massFracsRoe, gas) 		
	Cpi = calcCpMixture(massFracsRoe, gas)
	gammai = calcGammaMixture(Ri, Cpi)

	ci = np.sqrt(gammai * Ri * solPrimRoe[:,2])	# Roe average sound speed

	# if adapting pseudo time step later, compute maximum characteristic speed here
	if (params.adaptDTau):
		srf = np.maximum(solPrimRoe[:,1] + ci, solPrimRoe[:,1] - ci)
		sol.srf = np.maximum(srf[:-1], srf[1:])

	# dissipation term
	dQp = solPrimR - solPrimL
	M_ROE = calcRoeDissipation(solPrimRoe, rhoi, Hi, ci, Ri, Cpi, gas)
	dissTerm = 0.5 * (M_ROE * np.expand_dims(dQp, -2)).sum(-1)

	# complete Roe flux
	flux = 0.5 * (EL + ER) - dissTerm 

	return flux

# compute dissipation term of Roe flux
# inputs are all from Roe average state
def calcRoeDissipation(solPrim, rho, h0, c, R, Cp, gas: gasProps):

	# allocate
	dissMat = np.zeros((solPrim.shape[0], gas.numEqs, gas.numEqs), dtype = constants.realType)
	if (solPrim.dtype == constants.complexType):
		dissMat = np.zeros((solPrim.shape[0], gas.numEqs, gas.numEqs), dtype = constants.complexType)        
	
	# for clarity
	temp = solPrim[:,2]
	massFracs = solPrim[:,3:]

	rhoY = -np.square(rho) * (constants.RUniv * temp / solPrim[:,0] * gas.mwInvDiffs)
	hY = gas.enthRefDiffs + (temp - gas.tempRef) * gas.CpDiffs

	rhop = 1.0 / (R * temp) 			# derivative of density with respect to pressure
	rhoT = -rho / temp 					# derivative of density with respect to temperature
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
	dissMat[:,1,0] = u * phi_star + R_roe
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
def calcViscFlux(sol: solutionPhys, bounds: boundaries, params: parameters, gas: gasProps, geom: geometry, faceVals):

	# full domain state, including ghost cells
	if (params.spaceOrder == 1):
		solPrim = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim, bounds.outlet.sol.solPrim), axis = 0)
		rho = np.concatenate((bounds.inlet.sol.solCons[[0],0], sol.solCons[:,0], bounds.outlet.sol.solCons[[0],0]), axis = 0)
		cp = np.concatenate((bounds.inlet.sol.CpMix, sol.CpMix, bounds.outlet.sol.CpMix), axis = 0)

	elif (params.spaceOrder == 2):
		solPrim = faceVals[0]
		solConsFace = faceVals[1]
		rho = solConsFace[:,0]
		cp = faceVals[2]

	# convert to complex for complex step
	if (sol.solPrim.dtype == constants.complexType):
		rho = rho.astype(dtype=constants.complexType)
		cp = cp.astype(dtype=constants.complexType)


	solPrimGrad = np.zeros(solPrim.shape, dtype = constants.realType)
	if (solPrim.dtype == constants.complexType):
		solPrimGrad = np.zeros(solPrim.shape, dtype = constants.complexType)        

	# compute cell-centered gradients via finite difference stencil
	if (params.spaceOrder == 1):
		solPrimGrad[1:-1, :] = (0.5 / geom.dx) * (solPrim[2:, :] - solPrim[:-2, :])  
		solPrimGrad[0,:] = (solPrim[1,:] - solPrim[0,:]) / geom.dx       
		solPrimGrad[-1,:] = (solPrim[-1,:] - solPrim[-2,:]) / geom.dx         

	elif (params.spaceOrder == 2):
		solPrimGrad = faceVals[3]      

	# viscous flux vector
	Fv = np.zeros(solPrim.shape, dtype = constants.realType)
	if (sol.solPrim.dtype == constants.complexType):
		Fv = Fv.astype(dtype=constants.complexType)

	Ck = gas.muRef[:-1] * cp / gas.Pr[:-1] 								# thermal conductivity
	tau = 4.0/3.0 * gas.muRef[:-1] * solPrimGrad[:,1] 					# viscous stress "tensor"
	Fv[:,1] = Fv[:,1] + tau 											# finish momentum equation portion
	Fv[:,2] = Fv[:,2] + solPrim[:,1] * tau + Ck * solPrimGrad[:,2] 		# stress tensor component and thermal conductivity component of energy portion

	Cd = gas.muRef[:-1] / gas.Sc[:-1] / rho 							# mass diffusivity, from Schmidt number
	diff_rhoY = rho * Cd * np.squeeze(solPrimGrad[:,3:]) 				
	hY = gas.enthRefDiffs + (solPrim[:,2] - gas.tempRef) * gas.CpDiffs 	# enthalpy, by species
 
	if (gas.numSpecies > 1):
		Fv[:,2] = Fv[:,2] + np.sum(diff_rhoY * hY, axis = 1)			# complete mass diffusion component of viscous momentum flux 
		Fv[:,3:] = Fv[:,3:] + diff_rhoY 								# viscous scalar transport flux
	else:
		Fv[:,2] = Fv[:,2] + diff_rhoY * hY
		Fv[:,3] = Fv[:,3] + diff_rhoY
   	
	flux = 0.5 * (Fv[:-1,:] + Fv[1:,:]) 								# flux is average between adjacent cell centers?

	return flux

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

def reconstruct_2nd(sol: solutionPhys, bounds: boundaries, geom: geometry, gas: gasProps):
	
	#Gradients at all cell centres (including ghost) 
	Q = np.concatenate((bounds.inlet.sol.solPrim, sol.solPrim, bounds.outlet.sol.solPrim),axis=0)
	gradQ = np.zeros(Q.shape)
	gradQ[1:-1, :] =    (0.5 / geom.dx) * (Q[2:, :] - Q[:-2, :]) 
	gradQ[0, :]    =    (Q[1,:] - Q[0,:]) / geom.dx
	gradQ[-1,:]    =    (Q[-1,:] - Q[-2,:]) / geom.dx
	delQ           =    gradQ * (geom.dx / 2) #(grad * dx / 2) 
	
	#Max and min wrt neightbours at each cell
	Qln = np.concatenate(([Q[0, :]], Q[:-1, :]))
	Qm = Q
	Qrn = np.concatenate((Q[1:, :], [Q[-1, :]]))
	
	Qstack = np.stack((Qln, Qm, Qrn), axis=1)
	Qmax = np.amax(Qstack, axis=(0,1))
	Qmin = np.amin(Qstack, axis=(0,1))
	
	#Unconstrained reconstruction at each face
	Ql = Qm - delQ
	Qr = Qm + delQ
	
	#Gradient Limiting
	phil = np.ones(Q.shape)
	phir = np.ones(Q.shape)
	
	cond1l = (Ql - Qm > 0)
	cond1r = (Qr - Qm > 0)
	cond2l = (Ql - Qm < 0)
	cond2r = (Qr - Qm < 0)
	
	onesProf = np.ones(Q.shape)
	
	Qmax = Qmax * np.ones(Q.shape)
	Qmin = Qmin * np.ones(Q.shape)
	
	phil[cond1l] = np.minimum(onesProf[cond1l], (Qmax[cond1l] - Qm[cond1l]) / (Ql[cond1l] - Qm[cond1l]))
	phir[cond1r] = np.minimum(onesProf[cond1r], (Qmax[cond1r] - Qm[cond1r]) / (Qr[cond1r] - Qm[cond1r]))
	
	phil[cond2l] = np.minimum(onesProf[cond2l], (Qmin[cond2l] - Qm[cond2l]) / (Ql[cond2l] - Qm[cond2l]))
	phir[cond2r] = np.minimum(onesProf[cond2r], (Qmin[cond2r] - Qm[cond2r]) / (Qr[cond2r] - Qm[cond2r]))

	phi = np.minimum(phil, phir)
	
	Ql = Qm - phi * delQ
	Qr = Qm + phi * delQ
	
	solPrimL = Qr[:-1, :]
	solPrimR = Ql[1:, :]
	[solConsL, RMix, enthRefMix, CpMixL] = calcStateFromPrim(solPrimL, gas)
	[solConsR, RMix, enthRefMix, CpMixR] = calcStateFromPrim(solPrimR, gas)
	
	#Storing the face values
	solPrimFace = np.concatenate((solPrimL, solPrimR[[-1],:]), axis=0)
	solConsFace = np.concatenate((solConsL, solConsR[[-1],:]), axis=0)
	CpMixFace   = np.concatenate((CpMixL, CpMixR[[-1]]), axis=0)
	
	faceVals = []
	faceVals.append(solPrimFace)
	faceVals.append(solConsFace)
	faceVals.append(CpMixFace)
	faceVals.append(gradQ)
	
	return solPrimL, solConsL, solPrimR, solConsR, faceVals
	

	
		
	
		
		
	
	
	
	
	
   
	
	
	
	
	
		

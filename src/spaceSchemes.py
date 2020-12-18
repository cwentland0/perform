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

def writeToFile(fid, array):
	if (array.ndim > 1):
		array = array.flatten(order="F")
	fid.write(struct.pack('d'*array.shape[0], *(array)))

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

	# validate 
	fValid = open(os.path.join(params.workdir,"pyGEMSValOut.bin"),'wb')
	writeToFile(fValid, geom.xFace)
	writeToFile(fValid, solPrimL) 	# primitive state
	writeToFile(fValid, solPrimR)
	writeToFile(fValid, solPrimL) 	# this needs to be edited to output the higher-order version vs first order component
	writeToFile(fValid, solPrimR)
	writeToFile(fValid, solConsL[:,0]) # density
	writeToFile(fValid, solConsR[:,0])

	hYL = gas.enthRef + (np.repeat(solPrimL[:,[2]],2,axis=1) - gas.tempRef) * gas.Cp
	hYR = gas.enthRef + (np.repeat(solPrimR[:,[2]],2,axis=1) - gas.tempRef) * gas.Cp
	writeToFile(fValid, hYL)
	writeToFile(fValid, hYR)

	h0L = (solConsL[:,2] + solPrimL[:,0]) / solConsL[:,0]
	h0R = (solConsR[:,2] + solPrimR[:,0]) / solConsR[:,0]
	writeToFile(fValid, h0L)
	writeToFile(fValid, h0R)

	RL = stateFuncs.calcGasConstantMixture(solPrimL[:,3:], gas) # gas constant mixture
	RR = stateFuncs.calcGasConstantMixture(solPrimR[:,3:], gas)
	CpL = stateFuncs.calcCpMixture(solPrimL[:,3:], gas)
	CpR = stateFuncs.calcCpMixture(solPrimR[:,3:], gas)
	gammaL = stateFuncs.calcGammaMixture(RL, CpL)
	gammaR = stateFuncs.calcGammaMixture(RR, CpR)
	writeToFile(fValid, gammaL)
	writeToFile(fValid, gammaR)
	writeToFile(fValid, RL)
	writeToFile(fValid, RR)

	cL = np.sqrt(gammaL * RL * solPrimL[:,[2]])
	cR = np.sqrt(gammaR * RR * solPrimR[:,[2]])
	writeToFile(fValid, cL)
	writeToFile(fValid, cR)

	muL = gas.muRef[:-1] * np.ones(geom.numCells+1, dtype=np.float64)
	muR = gas.muRef[:-1] * np.ones(geom.numCells+1, dtype=np.float64)
	writeToFile(fValid, muL)
	writeToFile(fValid, muR)

	lambdaL = gas.muRef[:-1] * CpL / gas.Pr[:-1]
	lambdaR = gas.muRef[:-1] * CpR / gas.Pr[:-1]
	writeToFile(fValid, lambdaL)
	writeToFile(fValid, lambdaR)

	mdiL = gas.muRef / gas.Sc * np.ones((geom.numCells+1, gas.numSpeciesFull), dtype=np.float64)
	mdiR = gas.muRef / gas.Sc * np.ones((geom.numCells+1, gas.numSpeciesFull), dtype=np.float64)
	writeToFile(fValid, mdiL)
	writeToFile(fValid, mdiR)

	# compute fluxes
	flux, solPrimAve, solConsAve, CpAve = calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol, params, gas, fValid)
	if (params.viscScheme > 0):
		flux -= calcViscFlux(sol, solPrimAve, solConsAve, CpAve, bounds, params, gas, geom, faceVals, fValid)

	fValid.close()
	# pdb.set_trace()

	# compute RHS
	sol.RHS = flux[:-1,:] - flux[1:,:]
	sol.RHS[:,:] /= geom.dx	

	# compute source term
	if params.sourceOn:
		calcSource(sol, params, gas)
		sol.RHS[:,3:]  = sol.source + sol.RHS[:,3:]

# compute inviscid fluxes
# TODO: expand beyond Roe flux
# TODO: better naming conventions
# TODO: entropy fix
def calcInvFlux(solPrimL, solConsL, solPrimR, solConsR, sol: solutionPhys, params: parameters, gas: gasProps, fValid):

	# TODO: check for non-physical cells
	matShape = solPrimL.shape

	# inviscid flux vector
	EL = np.zeros(matShape, dtype = constants.realType)
	ER = np.zeros(matShape, dtype = constants.realType)

	if (solPrimL.dtype == constants.complexType):
		EL = np.zeros(matShape, dtype = constants.complexType)        
		ER = np.zeros(matShape, dtype = constants.complexType)        

	# compute sqrhol, sqrhor, fac, and fac1
	sqrhol = np.sqrt(solConsL[:, 0])
	sqrhor = np.sqrt(solConsR[:, 0])
	fac = sqrhol / (sqrhol + sqrhor)
	fac1 = 1.0 - fac

	h0L = stateFuncs.calcStagnationEnthalpy(solPrimL, gas)
	h0R = stateFuncs.calcStagnationEnthalpy(solPrimR, gas) 
	h0Ave = fac * h0L + fac1 * h0R 
	rhoAve = sqrhol * sqrhor
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

	if (params.adaptDTau):
		srf = np.maximum(solPrimAve[:,1] + cAve, solPrimAve[:,1] - cAve)
		sol.srf = np.maximum(srf[:-1], srf[1:])

	# dissipation term
	dQp = solPrimR - solPrimL
	M_ROE = calcRoeDissipation(solPrimAve, solConsAve[:,0], h0Ave, cAve, RAve, CpAve, gas)
	dissTerm = 0.5 * (M_ROE * np.expand_dims(dQp, -2)).sum(-1)

	# complete Roe flux
	flux = 0.5 * (EL + ER) - dissTerm 

	# validate
	writeToFile(fValid, solPrimAve) 	
	writeToFile(fValid, solConsAve[:,0])

	hYAve = gas.enthRef + (np.repeat(solPrimAve[:,[2]],2,axis=1) - gas.tempRef) * gas.Cp
	writeToFile(fValid, hYAve)

	h0Ave = (solConsAve[:,2] + solPrimAve[:,0]) / solConsAve[:,0]
	writeToFile(fValid, h0Ave)
	writeToFile(fValid, gammaAve)
	writeToFile(fValid, RAve)
	writeToFile(fValid, cAve)

	muAve = gas.muRef[:-1] * np.ones(EL.shape[0], dtype=np.float64)
	writeToFile(fValid, muAve)

	lambdaAve = gas.muRef[:-1] * CpAve / gas.Pr[:-1]
	writeToFile(fValid, lambdaAve)

	mdiAve = gas.muRef / gas.Sc * np.ones((EL.shape[0], gas.numSpeciesFull), dtype=np.float64)
	writeToFile(fValid, mdiAve)

	writeToFile(fValid, flux) 	# faceflux

	return flux, solPrimAve, solConsAve, CpAve

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
def calcViscFlux(sol: solutionPhys, solPrimAve, solConsAve, CpAve, bounds: boundaries, params: parameters, gas: gasProps, geom: geometry, faceVals, fValid):

	# compute state gradients
	solPrimGrad = np.zeros((geom.numCells+1, gas.numEqs), dtype = constants.realType)
	solPrimGrad[1:-1,:] = (sol.solPrim[1:, :] - sol.solPrim[:-1, :]) / geom.dx
	solPrimGrad[0,:] 	= (sol.solPrim[0, :] - bounds.inlet.sol.solPrim) / geom.dx 
	solPrimGrad[-1,:] 	= (bounds.outlet.sol.solPrim - sol.solPrim[-1,:]) / geom.dx

	Ck = gas.muRef[:-1] * CpAve / gas.Pr[:-1] 									# thermal conductivity
	tau = 4.0/3.0 * gas.muRef[:-1] * solPrimGrad[:,1] 							# stress "tensor"

	Cd = gas.muRef[:-1] / gas.Sc[:-1] / solConsAve[:,0]						# mass diffusivity
	diff_rhoY = solConsAve[:,0] * Cd * np.squeeze(solPrimGrad[:,3:])  		# 
	hY = gas.enthRefDiffs + (solPrimAve[:,2] - gas.tempRef) * gas.CpDiffs 	# species enthalpies, TODO: replace with stateFuncs function

	Fv = np.zeros((geom.numCells+1, gas.numEqs), dtype = constants.realType)
	Fv[:,1] = Fv[:,1] + tau 
	Fv[:,2] = Fv[:,2] + solPrimAve[:,1] * tau + Ck * solPrimGrad[:,2]
	if (gas.numSpecies > 1):
		Fv[:,2] = Fv[:,2] + np.sum(diff_rhoY * hY, axis = 1)
		Fv[:,3:] = Fv[:,3:] + diff_rhoY 
	else:
		Fv[:,2] = Fv[:,2] + diff_rhoY * hY
		Fv[:,3] = Fv[:,3] + diff_rhoY

	# validate
	writeToFile(fValid, solPrimGrad)
	writeToFile(fValid, Fv)

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
	[solConsL, RMix, enthRefMix, CpMixL] = stateFuncs.calcStateFromPrim(solPrimL, gas)
	[solConsR, RMix, enthRefMix, CpMixR] = stateFuncs.calcStateFromPrim(solPrimR, gas)
	
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
	

	
		
	
		
		
	
	
	
	
	
   
	
	
	
	
	
		

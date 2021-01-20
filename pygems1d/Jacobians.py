import pygems1d.constants as const

import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix, dia_matrix, diags
import pdb

# TODO: scipy is not strictly needed, should only import if it's available with a try, except.
#		In this case, can just make the Jacobian dense. It's a cost hit but makes code less restrictive.
# TODO: probably a lot of repeated calculations of full massFracs

def calcDSolPrimDSolCons(solInt):
	"""
	Compute the Jacobian of the conservative state w/r/t/ the primitive state
	This appears as \Gamma^{-1} in the pyGEMS documentation
	"""

	# TODO: some repeated calculations
	# TODO: add option for preconditioning dRho_dP

	gas = solInt.gasModel

	gammaMatrixInv = np.zeros((gas.numEqs, gas.numEqs, solInt.numCells))
	
	# for clarity
	rho       = solInt.solCons[0,:]
	press     = solInt.solPrim[0,:]
	vel       = solInt.solPrim[1,:]
	temp      = solInt.solPrim[2,:]
	massFracs = solInt.solPrim[3:,:]

	dRho_dP = solInt.dRho_dP; dRho_dT = solInt.dRho_dT; dRho_dY = solInt.dRho_dY
	dH_dP = solInt.dH_dP; dH_dT = solInt.dH_dT; dH_dY = solInt.dH_dY
	h0 = solInt.h0

	# some reused terms
	d = rho * dRho_dP * dH_dT - dRho_dT * (rho * dH_dP - 1.0)
	velSq = np.square(vel)

	# density row
	gammaMatrixInv[0,0,:] = (rho * dH_dT + dRho_dT * (h0 - velSq) + 
							np.sum(massFracs * (dRho_dY * dH_dT[None,:] - dRho_dT[None,:] * dH_dY), axis=0)) / d 
	gammaMatrixInv[0,1,:] = vel * dRho_dT / d
	gammaMatrixInv[0,2,:] = -dRho_dT / d
	gammaMatrixInv[0,3:,:] = (dRho_dT[None,:] * dH_dY - dRho_dY * dH_dT[None,:]) / d[None,:]
	
	# momentum row
	gammaMatrixInv[1,0,:] = -vel / rho
	gammaMatrixInv[1,1,:] = 1.0 / rho

	# energy row
	gammaMatrixInv[2,0,:] = (-dRho_dP * (h0 - velSq) - (rho * dH_dP - 1.0) + 
							np.sum(massFracs * ((rho * dRho_dP)[None,:] * dH_dY + dRho_dY * (rho * dH_dP - 1.0)[None,:]), axis=0) / rho) / d
	gammaMatrixInv[2,1,:] = -vel * dRho_dP / d
	gammaMatrixInv[2,2,:] = dRho_dP / d
	gammaMatrixInv[2,3:,:] = -((rho * dRho_dP)[None,:] * dH_dY + dRho_dY * (rho * dH_dP - 1.0)[None,:]) / (rho * d)[None,:]
	
	# species row(s)
	gammaMatrixInv[3:,0,:] = -massFracs / rho[None,:]
	for i in range(3,gas.numEqs):
		gammaMatrixInv[i,i,:] = 1.0 / rho
			
	return gammaMatrixInv


def calcDSolConsDSolPrim(solInt):
	"""
	Compute the Jacobian of conservative state w/r/t the primitive state
	This appears as \Gamma in the pyGEMS documentation
	"""

	# TODO: add option for preconditioning dRho_dP

	gas = solInt.gasModel

	gammaMatrix = np.zeros((gas.numEqs, gas.numEqs, solInt.numCells))

	# for clarity
	rho       = solInt.solCons[0,:]
	press     = solInt.solPrim[0,:]
	vel       = solInt.solPrim[1,:]
	temp      = solInt.solPrim[2,:]
	massFracs = solInt.solPrim[3:,:]

	dRho_dP = solInt.dRho_dP; dRho_dT = solInt.dRho_dT; dRho_dY = solInt.dRho_dY
	dH_dP = solInt.dH_dP; dH_dT = solInt.dH_dT; dH_dY = solInt.dH_dY
	h0 = solInt.h0	 
			
	# density row
	gammaMatrix[0,0,:]  = dRho_dP
	gammaMatrix[0,2,:]  = dRho_dT
	gammaMatrix[0,3:,:] = dRho_dY
		
	# momentum row
	gammaMatrix[1,0,:]  = vel * dRho_dP
	gammaMatrix[1,1,:]  = rho
	gammaMatrix[1,2,:]  = vel * dRho_dT
	gammaMatrix[1,3:,:] = vel[None,:] * dRho_dY
		
	# total energy row
	gammaMatrix[2,0,:]  = dRho_dP * h0 + rho * dH_dP - 1.0
	gammaMatrix[2,1,:]  = rho * vel
	gammaMatrix[2,2,:]  = dRho_dT * h0 + rho * dH_dT
	gammaMatrix[2,3:,:] = h0[None,:] * dRho_dY + rho[None,:] * dH_dY
	
	# species row
	gammaMatrix[3:,0,:] = massFracs[gas.massFracSlice,:] * dRho_dP[None,:]
	gammaMatrix[3:,2,:] = massFracs[gas.massFracSlice,:] * dRho_dT[None,:]
	for i in range(3, gas.numEqs):
		for j in range(3, gas.numEqs):
			gammaMatrix[i,j,:] = (i==j) * rho + massFracs[i-3,:] * dRho_dY[j-3,:]
				
	return gammaMatrix


def calcDSourceDSolPrim(solInt, dt):
	"""
	Compute source term Jacobian
	"""

	# TODO: does not account for reverse reaction, multiple reactions
	# TODO: does not account for temperature exponent in Arrhenius rate
	# TODO: really need to check that this works for more than a two-species reaction

	gas = solInt.gasModel

	dSdQp = np.zeros((gas.numEqs, gas.numEqs, solInt.numCells))

	rho       = solInt.solCons[0,:]
	press     = solInt.solPrim[0,:]
	temp      = solInt.solPrim[2,:]
	massFracs = solInt.solPrim[3:,:]
		
	dRho_dP = solInt.dRho_dP
	dRho_dT = solInt.dRho_dT
	dRho_dY = solInt.dRho_dY
	wf = solInt.wf

	specIdxs = np.squeeze(np.argwhere(gas.nuArr != 0.0))

	wf_divRho = np.sum(wf[None,:] * gas.nuArr[specIdxs,None], axis=0) / rho[None,:] 	# TODO: not correct for multi-reaction
	
	dWf_dT = wf_divRho * dRho_dT[None,:] - wf * gas.actEnergy / temp**2 	# negative, as activation energy already set as negative
	dWf_dP = wf_divRho * dRho_dP[None,:]

	# TODO: not correct for multi-reaction, this should be [numSpec, numReac, numCells] 
	dWf_dY = wf_divRho * dRho_dY
	for i in range(gas.numSpecies):
		posMFIdxs = np.nonzero(massFracs[i,:] > 0.0)[0]
		dWf_dY[i,posMFIdxs] += wf[posMFIdxs] * gas.nuArr[i] / massFracs[i,posMFIdxs]
	
	# TODO: for multi-reaction, should be a summation over the reactions here
	dSdQp[3:,0,:]  = -gas.molWeightNu[gas.massFracSlice,None] * dWf_dP
	dSdQp[3:,2,:]  = -gas.molWeightNu[gas.massFracSlice,None] * dWf_dT

	# TODO: this is totally wrong for multi-reaction
	for i in range(gas.numSpecies):
		dSdQp[3:,3+i,:] = -gas.molWeightNu[[i],None] * dWf_dY[[i],:]
	
	return dSdQp
	
   
def calcDInvFluxDSolPrim(sol):
	"""
	Compute Jacobian of inviscid flux vector with respect to primitive state
	Here, sol should be the solutionPhys associated with the left/right face state
	"""

	gas = sol.gasModel

	dFlux_dQp = np.zeros((gas.numEqs, gas.numEqs, sol.numCells))
	
	# for convenience
	rho       = sol.solCons[0,:]
	press     = sol.solPrim[0,:]
	vel       = sol.solPrim[1,:]
	velSq     = np.square(vel)
	temp      = sol.solPrim[2,:]
	massFracs = sol.solPrim[3:,:]
	h0        = sol.h0

	dRho_dP, dRho_dT, dRho_dY = gas.calcDensityDerivatives(rho,
								wrtPress=True, pressure=press,
								wrtTemp=True, temperature=temp,
								wrtSpec=True, massFracs=massFracs)

	dH_dP, dH_dT, dH_dY = gas.calcStagEnthalpyDerivatives(wrtPress=True,
							wrtTemp=True, massFracs=massFracs, 
							wrtSpec=True, temperature=temp)
	
	# continuity equation row
	dFlux_dQp[0,0,:]  = vel * dRho_dP
	dFlux_dQp[0,1,:]  = rho
	dFlux_dQp[0,2,:]  = vel * dRho_dT
	dFlux_dQp[0,3:,:] = vel[None,:] * dRho_dY
		
	# momentum equation row
	dFlux_dQp[1,0,:]  = dRho_dP * velSq + 1.0
	dFlux_dQp[1,1,:]  = 2.0 * rho * vel
	dFlux_dQp[1,2,:]  = dRho_dT * velSq
	dFlux_dQp[1,3:,:] = velSq[None,:] * dRho_dY
		
	# energy equation row
	dFlux_dQp[2,0,:]  = vel * (h0 * dRho_dP + rho * dH_dP)
	dFlux_dQp[2,1,:]  = rho * (velSq + h0)
	dFlux_dQp[2,2,:]  = vel * (h0 * dRho_dT + rho * dH_dT)
	dFlux_dQp[2,3:,:] = vel[None,:] * (h0[None,:] * dRho_dY + rho[None,:] * dH_dY)
	
	# species transport row(s)
	dFlux_dQp[3:,0,:] = massFracs * (dRho_dP * vel)[None,:]
	dFlux_dQp[3:,1,:] = massFracs * rho[None,:]
	dFlux_dQp[3:,2,:] = massFracs * (dRho_dT * vel)[None,:]
	# TODO: vectorize
	for i in range(3,gas.numEqs):
		for j in range(3,gas.numEqs):
			dFlux_dQp[i,j,:] = vel * ((i==j) * rho + massFracs[i-3,:] * dRho_dY[j-3,:])
			
	return dFlux_dQp


def calcDViscFluxDSolPrim(solAve):
	"""
	Compute Jacobian of viscous flux vector with respect to the primitive state
	solAve is the solutionPhys associated with the face state used to calculate the viscous flux
	"""

	gas = solAve.gasModel

	dFlux_dQp = np.zeros((gas.numEqs, gas.numEqs, solAve.numCells))

	# momentum equation row
	dFlux_dQp[1,1,:] = 4.0 / 3.0 * solAve.dynViscMix

	# energy equation row
	dFlux_dQp[2,1,:]  = 4.0 / 3.0 * solAve.solPrim[1,:] * solAve.dynViscMix
	dFlux_dQp[2,2,:]  = solAve.thermCondMix
	dFlux_dQp[2,3:,:] = solAve.solCons[[0],:] * (solAve.massDiffMix[gas.massFracSlice,:] * solAve.hi[gas.massFracSlice,:] - 
												 solAve.massDiffMix[[-1],:] * solAve.hi[[-1],:])

	# species transport row
	# TODO: vectorize
	for i in range(3, gas.numEqs):
		dFlux_dQp[i,i,:] = solAve.solCons[0,:] * solAve.massDiffMix[i-3,:]

	return dFlux_dQp


def calcDRoeFluxDSolPrim(solDomain, solver):
	"""
	Compute the gradient of the inviscid and viscous fluxes with respect to the primitive state
	"""

	RoeDiss = solDomain.RoeDiss.copy()

	dFluxL_dQpL = calcDInvFluxDSolPrim(solDomain.solL)
	dFluxR_dQpR = calcDInvFluxDSolPrim(solDomain.solR)

	if (solver.viscScheme > 0):
		dViscFlux_dQp = calcDViscFluxDSolPrim(solDomain.solAve)
		dFluxL_dQpL -= dViscFlux_dQp
		dFluxR_dQpR -= dViscFlux_dQp

	dFluxL_dQpL *= (0.5 / solver.mesh.dx)
	dFluxR_dQpR *= (0.5 / solver.mesh.dx)
	RoeDiss     *= (0.5 / solver.mesh.dx)

    # Jacobian wrt current cell
	dFlux_dQp = (dFluxL_dQpL[:,:,1:] + RoeDiss[:,:,1:]) + (-dFluxR_dQpR[:,:,:-1] + RoeDiss[:,:,:-1])
    
    # Jacobian wrt left neighbor
	dFlux_dQpL = (-dFluxL_dQpL[:,:,1:-1] - RoeDiss[:,:,:-2])
    
    # Jacobian wrt right neighbor
	dFlux_dQpR = (dFluxR_dQpR[:,:,1:-1] - RoeDiss[:,:,2:]) 
    	
	return dFlux_dQp, dFlux_dQpL, dFlux_dQpR


def calcDResDSolPrim(solDomain, solver):
	"""
	Compute Jacobian of the RHS function (i.e. fluxes and sources)  
	"""

	solInt = solDomain.solInt
	solIn  = solDomain.solIn
	solOut = solDomain.solOut
	gas    = solDomain.gasModel

	# stagnation enthalpy and derivatives of density and enthalpy, will need these regardless
	solInt.h0 = gas.calcStagnationEnthalpy(solInt.solPrim)
	solInt.dRho_dP, solInt.dRho_dT, solInt.dRho_dY = gas.calcDensityDerivatives(solInt.solCons[0,:],
													wrtPress=True, pressure=solInt.solPrim[0,:],
													wrtTemp=True, temperature=solInt.solPrim[2,:],
													wrtSpec=True, massFracs=solInt.solPrim[3:,:])

	solInt.dH_dP, solInt.dH_dT, solInt.dH_dY = gas.calcStagEnthalpyDerivatives(wrtPress=True,
												wrtTemp=True, massFracs=solInt.solPrim[3:,:], 
												wrtSpec=True, temperature=solInt.solPrim[2,:])


	# TODO: conditional path for other flux schemes
	# *_l is contribution to lower block diagonal, *_r is to upper block diagonal
	dFdQp, dFdQp_l, dFdQp_r = calcDRoeFluxDSolPrim(solDomain, solver)
	dRdQp = dFdQp.copy()

	# contribution to main block diagonal from source term Jacobian
	if solver.sourceOn:
		dSdQp = calcDSourceDSolPrim(solInt, solDomain.timeIntegrator.dt)
		dRdQp -= dSdQp

	# TODO: make this specific for each implicitIntegrator
	dtCoeffIdx = min(solver.iter, solDomain.timeIntegrator.timeOrder) - 1
	dtInv = solDomain.timeIntegrator.coeffs[dtCoeffIdx][0] / solDomain.timeIntegrator.dt

	# modifications depending on whether dual-time integration is being used
	if solDomain.timeIntegrator.dualTime:

		# contribution to main block diagonal from solution Jacobian
		gammaMatrix = calcDSolConsDSolPrim(solInt)
		if (solDomain.timeIntegrator.adaptDTau):
			dtauInv = calcAdaptiveDTau(solDomain, gammaMatrix)
		else:
			dtauInv = 1./solDomain.timeIntegrator.dtau

		dRdQp += gammaMatrix * (dtauInv[None,None,:] + dtInv)

		# assemble sparse Jacobian from main, upper, and lower block diagonals
		resJacob = resJacobAssemble(dRdQp, dFdQp_l, dFdQp_r)

	else:
		# TODO: this is hilariously inefficient, need to make Jacobian functions w/r/t conservative state
		#		convergence is also noticeably worse, since this is just approximate
		# transposes are due to matmul assuming stacks are in first index, maybe a better way to do this?
		gammaMatrixInv = np.transpose(calcDSolPrimDSolCons(solInt), axes=(2,0,1))
		dRdQ   = np.transpose(np.matmul(np.transpose(dRdQp, axes=(2,0,1)), gammaMatrixInv), axes=(1,2,0))
		dFdQ_l = np.transpose(np.matmul(np.transpose(dFdQp_l, axes=(2,0,1)), gammaMatrixInv[:-1,:,:]), axes=(1,2,0))
		dFdQ_r = np.transpose(np.matmul(np.transpose(dFdQp_r, axes=(2,0,1)), gammaMatrixInv[1:,:,:]), axes=(1,2,0))
		
		dtMat = np.repeat(dtInv*np.eye(gas.numEqs)[:,:,None], solInt.numCells, axis=2)
		dRdQ += dtMat

		resJacob = resJacobAssemble(dRdQ, dFdQ_l, dFdQ_r)

	return resJacob


def calcAdaptiveDTau(solDomain, gammaMatrix):
	"""
	Adapt dtau for each cell based on user input constraints and local wave speed
	"""

	# TODO: move this to implicitIntegrator

	# compute initial dtau from input CFL and srf (max characteristic speed)
	# srf is computed in calcInvFlux
	dtaum = 1.0/solDomain.solInt.srf
	dtau = solDomain.timeIntegrator.CFL*dtaum

	# limit by von Neumann number
	# TODO: THIS NU IS NOT CORRECT FOR A GENERAL MIXTURE
	nu = solDomain.gasModel.muRef[0] / solDomain.solInt.solCons[0,:]
	dtau = np.minimum(dtau, solDomain.timeIntegrator.VNN / nu)
	dtaum = np.minimum(dtaum, 3.0 / nu)

	# limit dtau
	# TODO: implement entirety of solutionChangeLimitedTimeStep from gems_precon.f90
	
	return  1.0 / dtau 

def resJacobAssemble(mat1, mat2, mat3):
	'''
	Reassemble residual Jacobian into a sparse 2D array for linear solve
	Stacking block diagonal forms of mat1 and block tri-diagonal form of mat2
	mat1 : 3-D Form of Gamma*(1/dt) + Gamma*(1/dtau) - dS/dQp + (dF/dQp)_i
	mat2 : 3-D Form of (dF/dQp)_(i-1) (Left Neighbor)
	mat3 : 3-D Form of (dF/dQp)_(i+1) (Right Neighbor)
	'''

	# TODO: this needs to be converted to C ordering, it causes headaches with the implicit ROMs

	numEqs, _, numCells = mat1.shape
	 
	# put arrays in proper format for use with bsr_matrix
	# zeroPad is because I don't know how to indicate that a row should have no blocks added when using bsr_matrix
	zeroPad = np.zeros((1,numEqs,numEqs), dtype = const.realType)
	center = np.transpose(mat1, (2,0,1))
	lower = np.concatenate((zeroPad, np.transpose(mat2, (2,0,1))), axis=0) 
	upper = np.concatenate((np.transpose(mat3, (2,0,1)), zeroPad), axis=0)

	# BSR format indices and indices pointers
	indptr = np.arange(numCells+1)
	indicesCenter = np.arange(numCells)
	indicesLower = np.arange(numCells)
	indicesLower[1:] -= 1
	indicesUpper = np.arange(1,numCells+1)
	indicesUpper[-1] -= 1

	# format center, lower, and upper block diagonals
	jacDim = numEqs * numCells
	centerSparse = bsr_matrix((center, indicesCenter, indptr), shape=(jacDim, jacDim))
	lowerSparse  = bsr_matrix((lower, indicesLower, indptr), shape=(jacDim, jacDim))
	upperSparse  = bsr_matrix((upper, indicesUpper, indptr), shape=(jacDim, jacDim))

	# assemble full matrix
	# convert to csr because spsolve requires this, I timed this and it's the most efficient way
	resJacob  = centerSparse + lowerSparse + upperSparse 
	resJacob = resJacob.tocsr(copy=False)

	return resJacob


# NOTE: this was an atrocious failure in an attempt to get resJacob into C-order
# easily 3-4 times more expensive that F-order above
# still would be really useful to get a C-order resJacob for ROMs
# def resJacobAssembleCOrder(centerBlock, lowerBlock, upperBlock):

# 	numEqs, _, numCells = centerBlock.shape
# 	jacDim = numEqs * numCells

# 	resJacob = csr_matrix((jacDim, jacDim), dtype=const.realType)

# 	for blockDiagIdx in range(numEqs):

# 		for varIdx1 in range(numEqs-blockDiagIdx):

# 			varIdx2 = varIdx1 + blockDiagIdx

# 			# center block diagonal
# 			if (blockDiagIdx == 0):
# 				if (varIdx1 == 0):
# 					center = centerBlock[varIdx1, varIdx2, :]	
# 					lower  = lowerBlock[varIdx1, varIdx2, :]
# 					upper  = upperBlock[varIdx1, varIdx2, :]
# 				else:
# 					center = np.concatenate((center, centerBlock[varIdx1, varIdx2, :]))
# 					lower  = np.concatenate((lower, np.array([0]), lowerBlock[varIdx1, varIdx2, :]))
# 					upper  = np.concatenate((upper, np.array([0]), upperBlock[varIdx1, varIdx2, :]))

# 			# off-center block diagonals
# 			else:

# 				if (varIdx1 == 0):
# 					upperCenter = centerBlock[varIdx1, varIdx2, :]
# 					upperLower  = np.concatenate((np.array([0]), lowerBlock[varIdx1, varIdx2, :]))
# 					upperUpper  = upperBlock[varIdx1, varIdx2, :]

# 					lowerCenter = centerBlock[varIdx2, varIdx1, :]
# 					lowerLower  = lowerBlock[varIdx2, varIdx1, :]
# 					lowerUpper  = np.concatenate((np.array([0]), upperBlock[varIdx2, varIdx1, :]))
# 				else:					
# 					upperCenter = np.concatenate((upperCenter, centerBlock[varIdx1, varIdx2, :]))
# 					upperLower  = np.concatenate((upperLower, np.array([0]), lowerBlock[varIdx1, varIdx2, :]))
# 					upperUpper  = np.concatenate((upperUpper, np.array([0]), upperBlock[varIdx1, varIdx2, :]))
				
# 					lowerCenter = np.concatenate((lowerCenter, centerBlock[varIdx2, varIdx1, :]))
# 					lowerLower  = np.concatenate((lowerLower, np.array([0]), lowerBlock[varIdx2, varIdx1, :]))
# 					lowerUpper  = np.concatenate((lowerUpper, np.array([0]), upperBlock[varIdx2, varIdx1, :]))

# 				if (varIdx1 == (numEqs-blockDiagIdx-1)):
# 					lowerUpper  = np.concatenate((lowerUpper, np.array([0])))
# 					upperLower  = np.concatenate((upperLower, np.array([0])))


# 		# center block diagonal
# 		if (blockDiagIdx == 0):
# 			resJacob = resJacob + diags((lower, center, upper), offsets=[-1,0,1], shape=(jacDim, jacDim), format="csr")

# 		# off-center block diagonals
# 		else:

# 			offset = blockDiagIdx * numCells
# 			resJacob += diags((lowerLower, lowerCenter, lowerUpper), offsets=[-offset-1,-offset,-offset+1], shape=(jacDim, jacDim), format="csr")
# 			resJacob += diags((upperLower, upperCenter, upperUpper), offsets=[offset-1,offset,offset+1], shape=(jacDim, jacDim), format="csr")

# 	return resJacob
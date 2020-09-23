from solution import solutionPhys, boundaries
from romClasses import solutionROM
from classDefs import parameters, geometry, gasProps
from spaceSchemes import calcRHS
from boundaryFuncs import calcBoundaries
from Jacobians import calcDResDSolPrim, calcDResDSolPrimImag
import constants
from scipy.sparse.linalg import spsolve
import numpy as np
import pdb

# initialize time history of solution
def initSolMat(sol, bounds, params, gas):
	
	sol_mat = []
	
	# calcBoundaries(sol, bounds, params, gas) # Ashish, why was this here?
	
	for _ in range(params.timeOrder+1):
		sol_mat.append(sol.solCons.copy())
	
	return sol_mat


# update time history of solution for implicit time integration
def updateSolMat(sol_mat):
	
	sol_mat[1:] = sol_mat[:-1]
			
	return sol_mat

# compute fully-discrete residual
# TODO: cold start is not valid for timeOrder > 2
def calcImplicitRes(sol: solutionPhys, sol_mat, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps, colstrt):
	
	t_or = params.timeOrder
	
	if (colstrt): # cold start
		params.timeOrder = 1 
	
	calcRHS(sol, bounds, params, geom, gas) # non-linear RHS of current solution
	
	if (params.timeOrder == 1):
		res = (sol.solCons - sol_mat[1])/(params.dt)
	elif (params.timeOrder == 2):
		res = (1.5*sol.solCons - 2.*sol_mat[1] + 0.5*sol_mat[2])/(params.dt)
	elif (params.timeOrder == 3):
		res = (11./6.*sol.solCons - 3.*sol_mat[1] + 1.5*sol_mat[2] -1./3.*sol_mat[3])/(params.dt)
	elif (params.timeOrder == 4):
		res = (25./12.*sol.solCons - 4.*sol_mat[1] + 3.*sol_mat[2] -4./3.*sol_mat[3] + 0.25*sol_mat[4])/(params.dt)
	else:
		raise ValueError("Implicit Schemes higher than BDF4 not-implemented")
	
	res = -res + sol.RHS
	params.timeOrder = t_or
	
	return res

# explicit time integrator, one subiteration
def advanceExplicit(sol: solutionPhys, rom: solutionROM, 
					bounds: boundaries, params: parameters, geom: geometry, gas: gasProps, 
					subiter, solOuter):
	
	
	#compute RHS function
	calcRHS(sol, bounds, params, geom, gas)
		
	# compute change in solution/code, advance solution/code
	if (params.calcROM):
		rom.mapRHSToModels(sol)
		rom.calcRHSProjection()
		rom.advanceSubiter(sol, params, subiter, solOuter)
	else:
		dSol = params.dt * params.subIterCoeffs[subiter] * sol.RHS
		sol.solCons = solOuter + dSol

	sol.updateState(gas)
	
	return sol
   
# implicit pseudo-time integrator, one subiteration
def advanceDual(sol, sol_mat, bounds, params, geom, gas, colstrt=False):
	
	# compute residual
	res = calcImplicitRes(sol, sol_mat, bounds, params, geom, gas, colstrt)
	
	# compute time/pseudo-time factors
	# TODO: add dynamic settings for this (i.e. robustness controls)
	dt_inv = constants.bdfCoeffs[params.timeOrder-1]/params.dt
	dtau_inv = 1./params.dtau

	# compute Jacobian or residual
	resJacob = calcDResDSolPrim(sol, gas, geom, params, bounds, dt_inv, dtau_inv)
	
	# Comparing with numerical jacobians
	# diff = calcDResDSolPrimImag(sol, gas, geom, params, bounds, dt_inv, dtau_inv)
	# print(diff)

	# solve linear system 
	dSol = spsolve(resJacob, res.flatten('C'))

	# update state
	sol.solPrim += dSol.reshape((geom.numCells, gas.numEqs), order='C')
	sol.updateState(gas, fromCons = False)

	# compute linear solve residual	
	res = resJacob @ dSol - res.flatten('C')
	
	return sol_mat, res

	
	
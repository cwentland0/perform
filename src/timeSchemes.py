from solution import solutionPhys, boundaries
from romClasses import solutionROM
from classDefs import parameters, geometry, gasProps
from spaceSchemes import calcRHS
from Jacobians import calcDResDSolPrim
import constants
from scipy.sparse.linalg import spsolve
import numpy as np
import pdb

   
# implicit pseudo-time integrator, one subiteration
def advanceDual(sol, bounds, params, geom, gas, subiter, colstrt=False):
	
	# non-linear RHS of current solution
	calcRHS(sol, bounds, params, geom, gas, subiter) 

	# compute residual
	calcImplicitRes(sol, bounds, params, geom, gas, subiter, colstrt)

	# compute Jacobian or residual
	resJacob = calcDResDSolPrim(sol, gas, geom, params, bounds)

	# solve linear system 
	dSol = spsolve(resJacob, sol.res.flatten('C'))

	# update state
	sol.solPrim += dSol.reshape((geom.numCells, gas.numEqs), order='C')
	sol.updateState(gas, fromCons = False)

	# update history
	sol.solHistCons[0] = sol.solCons.copy() 
	sol.solHistPrim[0] = sol.solPrim.copy() 

	# compute linear solve residual	
	sol.res = resJacob @ dSol - sol.res.flatten('C')
	
	
	
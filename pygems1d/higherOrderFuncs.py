from pygems1d.constants import realType

import numpy as np
import pdb

def calcCellGradients(solDomain, solver):
	
	# compute gradients via a finite difference stencil
	solPrimGrad = np.zeros(solDomain.solInt.solPrim.shape, dtype=realType)
	# TODO: this is not valid on a non-uniform grid
	# TODO: move this calculation to solutionDomain
	if (solver.spaceOrder == 2):
		solPrimGrad[:, 1:-1] = (0.5 / solver.mesh.dx) * (solDomain.solInt.solPrim[:,2:] - solDomain.solInt.solPrim[:,:-2])
		solPrimGrad[:, 0]    = (0.5 / solver.mesh.dx) * (solDomain.solInt.solPrim[:,1] - solDomain.solIn.solPrim[:,0])
		solPrimGrad[:, -1]   = (0.5 / solver.mesh.dx) * (solDomain.solOut.solPrim[:,0] - solDomain.solInt.solPrim[:,-2])
	else:
		raise ValueError("Order "+str(solver.spaceOrder)+" gradient calculations not implemented...")

	# compute gradient limiter and limit gradient, if necessary
	if (solver.gradLimiter > 0):

		# Barth-Jespersen
		if (solver.gradLimiter == 1):
			phi = limiterBarthJespersen(solDomain, solPrimGrad, solver.mesh)

		# Venkatakrishnan, no correction
		elif (solver.gradLimiter == 2):
			phi = limiterVenkatakrishnan(solDomain, solPrimGrad, solver.mesh)

		else:
			raise ValueError("Invalid input for gradLimiter: "+str(solver.gradLimiter))

		solPrimGrad = solPrimGrad * phi	# limit gradient

	return solPrimGrad
	
# find minimum and maximum of cell state and neighbor cell state
def findNeighborMinMax(solInterior, solInlet=None, solOutlet=None):

	# max and min of cell and neighbors
	solMax = solInterior.copy()
	solMin = solInterior.copy()

	# first compare against right neighbor
	solMax[:,:-1]  	= np.maximum(solInterior[:,:-1], solInterior[:,1:])
	solMin[:,:-1]  	= np.minimum(solInterior[:,:-1], solInterior[:,1:])
	if (solOutlet is not None):
		solMax[:,-1]	= np.maximum(solInterior[:,-1], solOutlet[:,0])
		solMin[:,-1]	= np.minimum(solInterior[:,-1], solOutlet[:,0])

	# then compare agains left neighbor
	solMax[:,1:] 	= np.maximum(solMax[:,1:], solInterior[:,:-1])
	solMin[:,1:] 	= np.minimum(solMin[:,1:], solInterior[:,:-1])
	if (solInlet is not None):
		solMax[:,0] 	= np.maximum(solMax[:,0], solInlet[:,0])
		solMin[:,0] 	= np.minimum(solMin[:,0], solInlet[:,0])

	return solMin, solMax

# Barth-Jespersen limiter
# ensures that no new minima or maxima are introduced in reconstruction
def limiterBarthJespersen(solDomain, grad, mesh):

	solPrim = solDomain.solInt.solPrim

	# get min/max of cell and neighbors
	solPrimMin, solPrimMax = findNeighborMinMax(solPrim, solDomain.solIn.solPrim, solDomain.solOut.solPrim)

	# unconstrained reconstruction at neighboring cell centers
	delSolPrim 		= grad * mesh.dx
	solPrimL 		= solPrim - delSolPrim
	solPrimR 		= solPrim + delSolPrim
	
	# limiter defaults to 1
	phiL = np.ones(solPrim.shape, dtype=realType)
	phiR = np.ones(solPrim.shape, dtype=realType)
	
	# find indices where difference in reconstruction is either positive or negative
	cond1L = ((solPrimL - solPrim) > 0)
	cond1R = ((solPrimR - solPrim) > 0)
	cond2L = ((solPrimL - solPrim) < 0)
	cond2R = ((solPrimR - solPrim) < 0)

	# threshold limiter for left and right reconstruction
	phiL[cond1L] = np.minimum(1.0, (solPrimMax[cond1L] - solPrim[cond1L]) / (solPrimL[cond1L] - solPrim[cond1L]))
	phiR[cond1R] = np.minimum(1.0, (solPrimMax[cond1R] - solPrim[cond1R]) / (solPrimR[cond1R] - solPrim[cond1R]))
	phiL[cond2L] = np.minimum(1.0, (solPrimMin[cond2L] - solPrim[cond2L]) / (solPrimL[cond2L] - solPrim[cond2L]))
	phiR[cond2R] = np.minimum(1.0, (solPrimMin[cond2R] - solPrim[cond2R]) / (solPrimR[cond2R] - solPrim[cond2R]))

	# take minimum limiter from left and right
	phi = np.minimum(phiL, phiR)
	
	return phi

# Venkatakrishnan limiter
# differentiable, but limits in uniform regions
def limiterVenkatakrishnan(solDomain, grad, mesh):

	solPrim = solDomain.solInt.solPrim

	# get min/max of cell and neighbors
	solPrimMin, solPrimMax = findNeighborMinMax(solPrim, solDomain.solIn.solPrim, solDomain.solOut.solPrim)

	# unconstrained reconstruction at neighboring cell centers
	delSolPrim 		= grad * mesh.dx
	solPrimL 		= solPrim - delSolPrim
	solPrimR 		= solPrim + delSolPrim
	
	# limiter defaults to 1
	phiL = np.ones(solPrim.shape, dtype=realType)
	phiR = np.ones(solPrim.shape, dtype=realType)
	
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

	# take minimum limiter from left and right
	phi = np.minimum(phiL, phiR)

	return phi
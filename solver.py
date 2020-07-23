import numpy as np
import matplotlib.pyplot as plt
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries
from boundaryFuncs import updateGhostCells
from spaceSchemes import calcRHS
import outputFuncs
import constants
import time
import os
import sys
import pdb

# driver function for advancing the solution
def solver(params: parameters, geom: geometry, gas: gasProps):

	# TODO: could move this to driver?
	# TODO: make an option to interpolate a solution onto the given mesh, if different
	# intialize solution
	initCond = np.load(params.initFile)
	sol = solutionPhys(geom.numCells, initCond[:,:,0], initCond[:,:,1], gas, params)
	sol.solPrim[:,1] += params.velAdd
	sol.updateState(gas, fromCons = False)
	
	# initialize boundary state
	bounds = boundaries(params, gas)

	# prep output
	fig, ax = plt.subplots(nrows=1, ncols=1)
	probeIdx = np.absolute(geom.x_cell - params.probeLoc).argmin()
	probeVals = np.zeros(params.numSteps, dtype = constants.floatType)
	tVals = np.linspace(params.dt, params.dt*params.numSteps, params.numSteps)
	if (params.visType == "field"):
		fieldImgDir = os.path.join(params.imgOutDir, "field_"+params.visVar+"_"+params.simType)
		if not os.path.isdir(fieldImgDir): os.mkdir(fieldImgDir)

	# loop over time iterations
	for tStep in range(params.numSteps):
		
		print("Iteration "+str(tStep))
		# call time integration scheme
		advanceSolution(sol, bounds, params, geom, gas)

		params.solTime += params.dt

		# write/store/visualize unsteady output
		if ( ((tStep+1) % params.outInterval) == 0):
			outputFuncs.storeFieldData(sol, params, tStep)
		outputFuncs.updateProbe(sol, params, probeVals, probeIdx, tStep)


		# draw visualization plots
		if ( ((tStep+1) % params.visInterval) == 0):
			if (params.visType == "field"): 
				outputFuncs.plotField(ax, sol, params, geom)
				if params.visSave: outputFuncs.writeFieldImg(fig, params, tStep, fieldImgDir)
			elif (params.visType == "probe"): 
				outputFuncs.plotProbe(ax, sol, params, probeVals, tStep, tVals)
			

	print("Solve finished, writing to disk")

	# write data to disk
	outputFuncs.writeData(sol, params, probeVals, tVals)

	# draw images, save to disk
	if ((params.visType == "probe") and params.visSave): 
		figFile = os.path.join(params.imgOutDir,"probe_"+params.visVar+"_"+params.simType+".png")
		fig.savefig(figFile)

# numerically integrate ODE forward one physical time step
# TODO: add implicit time integrators
def advanceSolution(sol: solutionPhys, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps):

	solConsOuter = sol.solCons.copy()

	# loop over max subiterations
	for subiter in range(params.numSubIters):

		# update boundary ghost cells
		updateGhostCells(sol, bounds, params, geom, gas)

		# compute RHS function
		calcRHS(sol, bounds, params, geom, gas)

		# if ROM, project onto test space

		# compute change in solution/code, advance solution/code
		dSolCons = params.dt * params.subIterCoeffs[subiter] * sol.RHS

		# if ROM, reconstruct solution

		# update state
		sol.solCons = solConsOuter + dSolCons
		sol.updateState(gas)

		# if implicit method, check residual and break if necessary

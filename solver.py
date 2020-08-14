import numpy as np
import matplotlib.pyplot as plt
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries, genInitialCondition
from boundaryFuncs import updateGhostCells
from spaceSchemes import calcRHS
from Jacobians import calc_dsolPrim,calc_dSourcedsolPrim,calc_dSourcedsolPrim_FD,calc_dSourcedsolPrim_imag
# from romClasses import solutionROM
from inputFuncs import readRestartFile
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
	# intialize from restart file
	if params.initFromRestart:
		params.solTime, solPrim0, solCons0 = readRestartFile(params.restOutDir)

	# otherwise init from scratch IC or custom IC file 
	else:
		if (params.initFile == None):
			solPrim0, solCons0 = genInitialCondition(params, gas, geom)
		else:
			# TODO: change this to .npz format with physical time included
			initCond = np.load(params.initFile)
			solPrim0 = initCond[:,:,0]
			solCons0 = initCond[:,:,0]
	sol = solutionPhys(geom.numCells, solPrim0, solCons0, gas, params)
	
	# add bulk velocity if required
	# TODO: should definitely be moved somewhere else
	if (params.velAdd > 0.0):
		sol.solPrim[:,1] += params.velAdd
	
	sol.updateState(gas, fromCons = False)

	# initialize ROM
	# if params.calcROM: rom = solutionROM(params.romInputs, sol, params)
	
	# initialize boundary state
	bounds = boundaries(sol, params, gas)

	# prep output
	fig, ax = plt.subplots(nrows=1, ncols=1)
	tVals = np.linspace(params.dt, params.dt*params.numSteps, params.numSteps)
	probeIdx = np.absolute(geom.x_cell - params.probeLoc).argmin()
	probeVals = np.zeros(params.numSteps, dtype = constants.floatType)
	if (params.visType == "field"):
		fieldImgDir = os.path.join(params.imgOutDir, "field_"+params.visVar+"_"+params.simType)
		if not os.path.isdir(fieldImgDir): os.mkdir(fieldImgDir)

	

	# loop over time iterations
	for tStep in range(params.numSteps):
		
		print("Iteration "+str(tStep+1))
		# call time integration scheme
		advanceSolution(sol, bounds, params, geom, gas)
		params.solTime += params.dt

		# write restart files
		if params.saveRestarts: 
			if ( ((tStep+1) % params.restartInterval) == 0):
				outputFuncs.writeRestartFile(sol, params, tStep)	 

		# write unsteady output
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

	return 

# numerically integrate ODE forward one physical time step
# TODO: add implicit time integrators
def advanceSolution(sol: solutionPhys, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps):


	if params.solforPrim:
		solOuter = sol.solPrim.copy()
	else:
		solOuter = sol.solCons.copy()        

	# loop over max subiterations
	for subiter in range(params.numSubIters):

		# update boundary ghost cells
		updateGhostCells(sol, bounds, params, geom, gas)

		# compute RHS function
		calcRHS(sol, bounds, params, geom, gas)

		# checking Source Term Jacobians
		#diff=calc_dSourcedsolPrim_imag(sol, gas, geom, params, params.dt, 1e-25)
        
        
        # if solPrim, calculate d(solPrim)/dt
		if params.solforPrim:        
  			sol.RHS=calc_dsolPrim(sol,gas) 

        # if ROM, project onto test space

		# compute change in solution/code, advance solution/code
		dSol = params.dt * params.subIterCoeffs[subiter] * sol.RHS

		# if ROM, reconstruct solution

		# update state
		if params.solforPrim:        
  			sol.solPrim = solOuter + dSol
  			sol.updateState(gas,fromCons=False)            
		else:      
  			sol.solCons = solOuter + dSol
  			sol.updateState(gas)
        
		# if implicit method, check residual and break if necessary

		
        

    
    
    



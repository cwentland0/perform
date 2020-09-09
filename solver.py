
import numpy as np
import matplotlib.pyplot as plt
import copy
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries, genInitialCondition
from boundaryFuncs import updateGhostCells
from spaceSchemes import calcRHS
#from Jacobians import calc_dsolPrim, calc_dSourcedsolPrim, calc_dSourcedsolPrim_FD, calc_dSourcedsolPrim_imag
from timeSchemes import advanceexplicit, advancedual, init_sol_mat, update_sol_mat
# from romClasses import solutionROM
from inputFuncs import readRestartFile
import outputFuncs
import constants
import time
import os
import sys
import pdb
#profiling
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

# driver function for advancing the solution
#@profile
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
		if (params.solTime == 0):  
   			
   			sol_mat = init_sol_mat(sol, bounds, params, geom, gas) #initialising time-memory

		sol_mat = advanceSolution(sol, bounds, params, geom, gas, sol_mat)

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


# numerically integrate ODE forward one physical time step
# TODO: add implicit time integrators
#@profile
def advanceSolution(sol: solutionPhys, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps, sol_mat):

	solOuter = sol.solCons.copy()
    

	# loop over max subiterations
	for subiter in range(params.numSubIters):
      
        # update boundary ghost cells
        
		if (params.timeType == 'explicit'):  
   			updateGhostCells(sol, bounds, params, geom, gas)
   			sol = advanceexplicit(sol, bounds, params, geom, gas, subiter, solOuter)
                       
		else:  

			if (params.solTime <= params.timeOrder*params.dt): 
				sol_mat,res = advancedual(sol, sol_mat, bounds, params, geom, gas, colstrt=True) #cold-start
				updateGhostCells(sol, bounds, params, geom, gas)           
				sol_mat[0] = sol.solCons.copy()           
			else:
				sol_mat,res = advancedual(sol, sol_mat, bounds, params, geom, gas)
				updateGhostCells(sol, bounds, params, geom, gas)           
				sol_mat[0] = sol.solCons.copy()     
       			 
		if (params.timeType == 'implicit'):  
			#print(np.linalg.norm(res,ord=2)) #printitng sub-iterations convergence
			if (np.linalg.norm(res,ord=2) < params.resTol): 
				break

	if(params.timeType == 'implicit'):
		
		sol_mat = update_sol_mat(sol_mat, bounds, params, geom, gas) #updating time-memory
		return sol_mat
      
             

		
        

    
    
    



import argparse
import constants
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # don't print all the TensorFlow warnings
from systemSolver import systemSolver
from solution import solutionPhys, boundaries
from inputFuncs import getInitialConditions
import outputFuncs
import numpy as np
import traceback
import pdb

# TODO: check for incorrect assignments that need a copy instead
# TODO: rename sol variable used to represent full domain state, our boundary sol name. Is confusing
# TODO: make code general for more than two species, array broadcasts are different for 2 vs 3+ species
#		idea: custom iterators for species-related slicing, or just squeeze any massfrac references

def main():

	##### START SOLVER SETUP #####

	# read working directory input
	parser = argparse.ArgumentParser(description = "Read working directory")
	parser.add_argument('workingDir', type = str, default = "./", help="runtime working directory")
	args = parser.parse_args()
	constants.workingDir = args.workingDir
	constants.workingDir = os.path.expanduser(constants.workingDir)

	# make output directories
	# TODO: shove this into a function
	constants.unsteadyOutputDir = os.path.join(constants.workingDir, constants.unsteadyOutputDirName)
	if not os.path.isdir(constants.unsteadyOutputDir): 	os.mkdir(constants.unsteadyOutputDir)
	constants.probeOutputDir = os.path.join(constants.workingDir, constants.probeOutputDirName)
	if not os.path.isdir(constants.probeOutputDir): 	os.mkdir(constants.probeOutputDir)
	constants.imageOutputDir = os.path.join(constants.workingDir, constants.imageOutputDirName)
	if not os.path.isdir(constants.imageOutputDir): 	os.mkdir(constants.imageOutputDir)
	constants.restartOutputDir = os.path.join(constants.workingDir, constants.restartOutputDirName)
	if not os.path.isdir(constants.restartOutputDir): 	os.mkdir(constants.restartOutputDir)

	# setup solver(s)
	# TODO: multi-domain solvers
	solver = systemSolver()

	##### END SOLVER SETUP #####

	##### START SOLUTION INITIALIZATION #####

	# initialize unsteady solution, boundary state, and ROM state
	solPrim0, solCons0 = getInitialConditions(solver)
	sol = solutionPhys(solPrim0, solCons0, solver.mesh.numCells, solver)
	bounds = boundaries(sol, solver)
	if solver.calcROM: 
		rom = solutionROM(solver.romInputs, sol, solver)
		rom.initializeROMState(sol)
	else:
		rom = None

	##### END SOLUTION INITIALIZATION #####

	##### START VISUALIZATION SETUP #####
	# TODO: clean this junk up

	# prep probe
	# TODO: expand to multiple probe locations
	probeIdx = 0
	if (solver.probeLoc > solver.mesh.xR):
		solver.probeSec = "outlet"
	elif (solver.probeLoc < solver.mesh.xL):
		solver.probeSec = "inlet"
	else:
		solver.probeSec = "interior"
		probeIdx = np.absolute(solver.mesh.xCell - solver.probeLoc).argmin()
	probeVals = np.zeros((solver.timeIntegrator.numSteps, solver.numVis), dtype = constants.realType)

	if (solver.visType != "None"): 
		fig, ax, axLabels = outputFuncs.setupPlotAxes(solver)
		visName = ""
		for visVar in solver.visVar:
			visName += "_"+visVar
		visName += "_"+solver.simType

	tVals = np.linspace(solver.timeIntegrator.dt,
						solver.timeIntegrator.dt*solver.timeIntegrator.numSteps, 
						solver.timeIntegrator.numSteps, dtype = constants.realType)
	if ((solver.visType == "field") and solver.visSave):
		fieldImgDir = os.path.join(solver.imgOutDir, "field"+visName)
		if not os.path.isdir(fieldImgDir): os.mkdir(fieldImgDir)
	else:
		fieldImgDir = None

	##### END VISUALIZATION SETUP #####

	##### START UNSTEADY SOLUTION #####

	# loop over time iterations
	for solver.timeIntegrator.iter in range(1,solver.timeIntegrator.numSteps+1):
		
		# advance one physical time step
		solver.timeIntegrator.advanceIter(sol, rom, bounds, solver)
		solver.solTime += solver.timeIntegrator.dt

		# write restart files
		if solver.saveRestarts: 
			if ( (solver.timeIntegrator.iter % solver.restartInterval) == 0):
				outputFuncs.writeRestartFile(sol, solver)	 

		# write output
		if (( solver.timeIntegrator.iter % solver.outInterval) == 0):
			outputFuncs.storeFieldDataUnsteady(sol, solver)
		outputFuncs.updateProbe(sol, solver, bounds, probeVals, probeIdx)

		# "steady" solver processing
		if (solver.timeIntegrator.runSteady):
			sol.resOutput(solver)
			if ((solver.timeIntegrator.iter % solver.outInterval) == 0): 
				outputFuncs.writeDataSteady(sol, solver)
			outputFuncs.updateResOut(sol, solver)
			if (sol.resOutL2 < solver.steadyThresh): 
				print("Steady residual criterion met, terminating run...")
				break 	# quit if steady residual threshold met

		# draw visualization plots
		if ( (solver.timeIntegrator.iter % solver.visInterval) == 0):
			if (solver.visType == "field"): 
				outputFuncs.plotField(fig, ax, axLabels, sol, solver)
				if solver.visSave: outputFuncs.writeFieldImg(fig, solver, fieldImgDir)
			elif (solver.visType == "probe"): 
				outputFuncs.plotProbe(fig, ax, axLabels, sol, solver, probeVals, tVals)
			
	print("Solve finished, writing to disk!")

	##### END UNSTEADY SOLUTION #####

	##### START POST-PROCESSING #####

	# write data to disk
	outputFuncs.writeDataUnsteady(sol, solver, probeVals, tVals)
	if (solver.timeIntegrator.runSteady): outputFuncs.writeDataSteady(sol, solver)

	# draw final images, save to disk
	if ((solver.visType == "probe") and solver.visSave): 
		figFile = os.path.join(solver.imgOutDir,"probe"+visName+".png")
		fig.savefig(figFile)

	##### END POST-PROCESSING #####

if __name__ == "__main__":
	try:
		main()
		print("Run finished!")
	except:
		print(traceback.format_exc())
		print("Run failed!")
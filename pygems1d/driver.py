import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # don't print all the TensorFlow warnings
import pygems1d.constants as const
from pygems1d.systemSolver import systemSolver
from pygems1d.solution.solutionDomain import solutionDomain
from pygems1d.miscFuncs import mkdirInWorkdir
import pygems1d.outputFuncs as outputFuncs
import numpy as np
import argparse
import traceback
import pdb

# TODO: check for incorrect assignments that need a copy instead
# TODO: make code general for more than two species, array broadcasts are different for 2 vs 3+ species
#		idea: custom iterators for species-related slicing, or just squeeze any massfrac references

def main():

	##### START SOLVER SETUP #####

	# read working directory input
	parser = argparse.ArgumentParser(description = "Read working directory")
	parser.add_argument('workingDir', type = str, default = "./", help="runtime working directory")
	args = parser.parse_args()
	const.workingDir = args.workingDir
	const.workingDir = os.path.expanduser(const.workingDir)
	assert (os.path.isdir(const.workingDir)), "Given working directory does not exist"

	# make output directories
	const.unsteadyOutputDir = mkdirInWorkdir(const.unsteadyOutputDirName)
	const.probeOutputDir = mkdirInWorkdir(const.probeOutputDirName)
	const.imageOutputDir = mkdirInWorkdir(const.imageOutputDirName)
	const.restartOutputDir = mkdirInWorkdir(const.restartOutputDirName)

	# setup solver(s)
	# TODO: multi-domain solvers
	solver = systemSolver()

	##### END SOLVER SETUP #####

	##### START SOLUTION INITIALIZATION #####

	# initialize interior and boundary state, and ROM state
	
	solDomain = solutionDomain(solver)

	if solver.calcROM: 
		solROM = solutionROM(solver.romInputs, solDomain.solInt, solver)
		solROM.initializeROMState(solDomain.solInt)
	else:
		solROM = None

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
	probeVals = np.zeros((solver.timeIntegrator.numSteps, solver.numVis), dtype = const.realType)

	if (solver.visType != "None"): 
		fig, ax, axLabels = outputFuncs.setupPlotAxes(solver)
		visName = ""
		for visVar in solver.visVar:
			visName += "_"+visVar
		visName += "_"+solver.simType

	tVals = np.linspace(solver.timeIntegrator.dt,
						solver.timeIntegrator.dt*solver.timeIntegrator.numSteps, 
						solver.timeIntegrator.numSteps, dtype = const.realType)
	if ((solver.visType == "field") and solver.visSave):
		fieldImgDir = os.path.join(solver.imgOutDir, "field"+visName)
		if not os.path.isdir(fieldImgDir): os.mkdir(fieldImgDir)
	else:
		fieldImgDir = None

	##### END VISUALIZATION SETUP #####

	##### START UNSTEADY SOLUTION #####

	# loop over time iterations
	for solver.timeIntegrator.iter in range(1, solver.timeIntegrator.numSteps+1):
		
		# advance one physical time step
		solver.timeIntegrator.advanceIter(solDomain, solROM, solver)
		solver.solTime += solver.timeIntegrator.dt

		# write restart files
		if solver.saveRestarts: 
			if ( (solver.timeIntegrator.iter % solver.restartInterval) == 0):
				outputFuncs.writeRestartFile(solDomain.solInt, solver)	 

		# write output
		if (( solver.timeIntegrator.iter % solver.outInterval) == 0):
			outputFuncs.storeFieldDataUnsteady(solDomain.solInt, solver)
		outputFuncs.updateProbe(solDomain, solver, probeVals, probeIdx)

		# "steady" solver processing
		if (solver.timeIntegrator.runSteady):
			if ((solver.timeIntegrator.iter % solver.outInterval) == 0): 
				outputFuncs.writeDataSteady(solDomain.solInt, solver)
			outputFuncs.updateResOut(solDomain.solInt, solver)
			if (solDomain.solInt.resNormL2 < solver.steadyThresh): 
				print("Steady residual criterion met, terminating run")
				break 	# quit if steady residual threshold met

		# draw visualization plots
		if ( (solver.timeIntegrator.iter % solver.visInterval) == 0):
			if (solver.visType == "field"): 
				outputFuncs.plotField(fig, ax, axLabels, solDomain.solInt, solver)
				if solver.visSave: outputFuncs.writeFieldImg(fig, solver, fieldImgDir)
			elif (solver.visType == "probe"): 
				outputFuncs.plotProbe(fig, ax, axLabels, solver, probeVals, tVals)
			
	print("Solve finished, writing to disk!")

	##### END UNSTEADY SOLUTION #####

	##### START POST-PROCESSING #####

	# write data to disk
	outputFuncs.writeDataUnsteady(solDomain.solInt, solver, probeVals, tVals)
	if (solver.timeIntegrator.runSteady): 
		outputFuncs.writeDataSteady(solDomain.solInt, solver)

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
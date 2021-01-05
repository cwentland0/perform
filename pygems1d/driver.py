import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # don't print all the TensorFlow warnings
import pygems1d.constants as const
from pygems1d.systemSolver import systemSolver
from pygems1d.solution.solutionDomain import solutionDomain
from pygems1d.miscFuncs import mkdirInWorkdir
from pygems1d.visualization.visualizationGroup import visualizationGroup

import numpy as np
import argparse
import traceback
import warnings
warnings.filterwarnings("error")
from time import time
import pdb

# TODO: find some way to untangle circular dependencies in systemSolver 
# TODO: make code general for more than two species, array broadcasts are different for 2 vs 3+ species
#		idea: custom iterators for species-related slicing, or just squeeze any massfrac references

def main():

	##### START SETUP #####

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

	solver = systemSolver()				# mesh, gas model, and time integrator
										# TODO: multi-domain solvers

	solDomain = solutionDomain(solver)	# physical solution

	# ROM definition and solution
	if solver.calcROM: 
		solROM = solutionROM(solver.romInputs, solDomain.solInt, solver)
		solROM.initializeROMState(solDomain.solInt)
	else:
		solROM = None

	visGroup = visualizationGroup(solver) # plots

	##### END SETUP #####

	##### START UNSTEADY SOLUTION #####

	try:
		# loop over time iterations
		t1 = time()
		for solver.timeIntegrator.iter in range(1, solver.timeIntegrator.numSteps+1):
			
			# advance one physical time step
			solver.timeIntegrator.advanceIter(solDomain, solROM, solver)

			# write unsteady solution outputs
			solDomain.writeIterOutputs(solver)

			# check "steady" solve
			if (solver.timeIntegrator.runSteady):
				breakFlag = solDomain.writeSteadyOutputs(solver)
				if breakFlag: break

			# visualization
			visGroup.drawPlots(solDomain, solver)

		runtime = time() - t1
		print("Solve finished in %.8f seconds, writing to disk" % runtime)

	except RuntimeWarning:
		solver.solveFailed = True
		print(traceback.format_exc())
		print("Solve failed, dumping solution so far to disk")

	##### END UNSTEADY SOLUTION #####

	##### START POST-PROCESSING #####

	solDomain.writeFinalOutputs(solver)
	# visGroup.savePlots(solver)

	# # draw final images, save to disk
	# if ((solver.visType == "probe") and solver.visSave): 
	# 	figFile = os.path.join(solver.imgOutDir,"probe"+visName+".png")
	# 	fig.savefig(figFile)

	##### END POST-PROCESSING #####

if __name__ == "__main__":
	try:
		main()
	except:
		print(traceback.format_exc())
		print("Execution failed")
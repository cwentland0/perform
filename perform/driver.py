import os
from time import time
import argparse
import traceback
import warnings
warnings.filterwarnings("error")

import perform.constants as const
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.visualization_group import VisualizationGroup
from perform.rom.rom_domain import RomDomain
from perform.misc_funcs import mkdir_in_workdir

# TODO: Check all calls to calcDensityDerivatives and calcStagEnthalpyDerivatives, pass full massFrac array
# TODO: ^^^^^ Emphasizing this, this is a big contribution to Jacobian cost


def main():

	##### Start setup #####

	# Read working directory input
	parser = argparse.ArgumentParser(description = "Read working directory")
	parser.add_argument('working_dir', type = str, default = "./", help="runtime working directory")
	const.working_dir = os.path.expanduser(parser.parse_args().working_dir)
	assert (os.path.isdir(const.working_dir)), "Given working directory does not exist"

	# Make output directories
	const.unsteady_output_dir = mkdir_in_workdir(const.unsteady_output_dir_name)
	const.probe_output_dir = mkdir_in_workdir(const.probeOutputDirName)
	const.image_output_dir = mkdir_in_workdir(const.imageOutputDirName)
	const.restart_output_dir = mkdir_in_workdir(const.restartOutputDirName)

	# Retrieve solver parameters and initialize mesh
	# TODO: multi-domain solvers
	# TODO: move mesh to solution_domain
	solver = SystemSolver()				
										
	# Initialize physical and ROM solutions
	sol_domain = SolutionDomain(solver)	
	if solver.calcROM:
		rom_domain = RomDomain(sol_domain, solver)
	else:
		rom_domain = None

	# Initialize plots
	visGroup = VisualizationGroup(solDomain, solver) 

	##### End setup #####

	##### Start unsteady solution #####

	try:
		# Loop over time iterations
		time_start = time()
		for solver.iter in range(1, solver.numSteps+1):
			
			# advance one physical time step
			if (solver.calcROM):
				rom_domain.advance_iter(sol_domain, solver)
			else:
				sol_domain.advance_iter(solver)
			solver.time_iter += 1
			solver.sol_time  += solver.dt

			# write unsteady solution outputs
			sol_domain.write_iter_outputs(solver)

			# check "steady" solve
			if solver.run_steady:
				break_flag = sol_domain.write_steady_outputs(solver)
				if break_flag: break

			# visualization
			visGroup.draw_plots(sol_domain, solver)

		runtime = time() - time_start
		print("Solve finished in %.8f seconds, writing to disk" % runtime)

	except RuntimeWarning:
		solver.solve_failed = True
		print(traceback.format_exc())
		print("Solve failed, dumping solution so far to disk")

	##### End unsteady solution #####

	##### Start post-processing #####

	solDomain.write_final_outputs(solver)

	##### End post-processing #####

if __name__ == "__main__":
	try:
		main()
	except:
		print(traceback.format_exc())
		print("Execution failed")
"""Driver for executing PERFORM simulations.

Initializes all necessary constructs for executing a PERFORM simulation,
namely a SystemSolver, a SolutionDomain, a VisualizationGroup, and a RomDomain (if running a ROM simulation).
Advances through time steps, calls visualization and output routines, and handles solver blowup if it occurs.

After installing PERFORM, the terminal command "perform" will execute main() and take the first command line
argument as the working directory
"""

import os
from time import time
import argparse
import traceback
import warnings

from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.visualization_group import VisualizationGroup
from perform.rom.rom_domain import RomDomain

warnings.filterwarnings("error")


def driver(working_dir):
    """Main driver function which initializes all necessary constructs and advances the solution in time"""

    # ----- Start setup -----

    working_dir = os.path.expanduser(working_dir)
    assert os.path.isdir(working_dir), "Given working directory does not exist"

    # Retrieve global solver parameters
    # TODO: multi-domain solvers
    solver = SystemSolver(working_dir)

    # Initialize physical and ROM solutions
    sol_domain = SolutionDomain(solver)
    if solver.calc_rom:
        rom_domain = RomDomain(sol_domain, solver)
    else:
        rom_domain = None

    # Initialize plots
    visGroup = VisualizationGroup(sol_domain, solver)

    # ----- End setup -----

    # ----- Start unsteady solution -----

    try:
        # Loop over time iterations
        time_start = time()
        for solver.iter in range(1, solver.num_steps + 1):

            # Advance one physical time step
            if solver.calc_rom:
                rom_domain.advance_iter(sol_domain, solver)
            else:
                sol_domain.advance_iter(solver)
            solver.time_iter += 1
            solver.sol_time += solver.dt

            # Write unsteady solution outputs
            sol_domain.write_iter_outputs(solver)

            # Check "steady" solve
            if solver.run_steady:
                break_flag = sol_domain.write_steady_outputs(solver)
                if break_flag:
                    break

            # Visualization
            visGroup.draw_plots(sol_domain, solver)

        runtime = time() - time_start
        print("Solve finished in %.8f seconds, writing to disk" % runtime)

    except RuntimeWarning:
        solver.solve_failed = True
        print(traceback.format_exc())
        print("Solve failed, dumping solution so far to disk")

    # ----- End unsteady solution -----

    # ----- Start post-processing -----

    sol_domain.write_final_outputs(solver)

    # ----- End post-processing -----


def main():

    # Read working directory input
    parser = argparse.ArgumentParser(description="Read working directory")
    parser.add_argument("working_dir", type=str, default="./", help="runtime working directory")
    working_dir = parser.parse_args().working_dir

    driver(working_dir)


if __name__ == "__main__":

    try:
        main()
    except:
        print(traceback.format_exc())
        print("Execution failed")

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
import sys

from perform.constants import PARAM_INPUTS, ROM_INPUTS
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.visualization_group import VisualizationGroup
from perform.rom.rom_domain import RomDomain

warnings.filterwarnings("error")

def driver(working_dir, params_inp=PARAM_INPUTS, rom_inp=ROM_INPUTS):
    """Main driver function which initializes all necessary constructs and advances the solution in time"""

    # ----- Start setup -----

    working_dir = os.path.expanduser(working_dir)
    assert os.path.isdir(working_dir), "Given working directory does not exist"

    # Retrieve global solver parameters
    # TODO: multi-domain solvers
    solver = SystemSolver(working_dir, params_inp, rom_inp)

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

        # run RHS calculation and update solution probes to store IC data correctly
        solver.iter = 0
        sol_domain.update_probes(solver, is_sol_data=True)
        sol_domain.calc_rhs(solver)

        # Loop over time iterations
        time_start = time()
        for solver.iter in range(1, solver.num_steps + 1):

            # Advance one physical time step
            if solver.calc_rom:
                rom_domain.advance_time(sol_domain, solver)
            else:
                sol_domain.advance_iter(solver)
            solver.time_iter += 1
            solver.sol_time += solver.dt

            # Write unsteady solution outputs
            sol_domain.write_sol_outputs(solver)

            # Check "steady" solve
            if solver.run_steady:
                break_flag = sol_domain.write_steady_outputs(solver)
                if break_flag:
                    break

            # Visualization
            visGroup.draw_plots(sol_domain, solver)

        runtime = time() - time_start
        if solver.stdout:
            print("Solve finished in %.8f seconds, writing to disk" % runtime)

    except RuntimeWarning:
        solver.solve_failed = True
        print(traceback.format_exc())
        print("Solve failed, dumping solution so far to disk")

    # ----- End unsteady solution -----

    # ----- Start post-processing -----

    # Run RHS calculations one more time to get non-solution data at final snapshot, if necessary
    # TODO: jank, could just pass an iteration number to write_nonsol_outputs
    if not solver.solve_failed:
        sol_domain.calc_rhs(solver)
    solver.iter += 1
    sol_domain.write_nonsol_outputs(solver)
    solver.iter -= 1

    if rom_domain == None:
        sol_domain.write_final_outputs(solver)
    else:
        sol_domain.write_final_outputs(solver, rom_domain.param_string)

    # ----- End post-processing -----


def main():

    # Read working directory, solver params, rom params input
    parser = argparse.ArgumentParser(description="Read working directory")
    parser.add_argument("-w", "--work", type=str, default="./", help="runtime working directory")
    parser.add_argument(
        "-p", "--param", type=str, default="./" + PARAM_INPUTS, help="solver parameters input file path"
    )
    parser.add_argument("-r", "--rom", type=str, default="./" + ROM_INPUTS, help="ROM parameter input file path")
    working_dir = parser.parse_args().work
    params_inp = parser.parse_args().param
    rom_inp = parser.parse_args().rom

    driver(working_dir, params_inp, rom_inp)


if __name__ == "__main__":

    try:
        main()
    except:
        print(traceback.format_exc())
        print("Execution failed")

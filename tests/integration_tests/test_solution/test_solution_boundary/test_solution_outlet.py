import unittest
import os
import shutil

import numpy as np

from constants import CHEM_DICT_REACT
import perform.constants as constants
from perform.system_solver import SystemSolver
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_phys import SolutionPhys 
from perform.solution.solution_boundary.solution_outlet import SolutionOutlet


class SolutionOutletMethodTests(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        # set chemistry
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # NOTE: press, vel, temp, and rho set later, depending on boundary condition
        self.bound_type = "outlet"
        self.mass_fracs = [0.6, 0.4]
        self.pert_type = "pressure"
        self.pert_perc = 0.025
        self.pert_freq = [125e3]

        # generate file with necessary input files
        self.test_file = os.path.join(self.test_dir, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-7\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

            f.write("mass_fracs_" + self.bound_type + " = " + str(self.mass_fracs) + "\n")
            f.write("pert_type_" + self.bound_type + " = \"" + str(self.pert_type) + "\"\n")
            f.write("pert_perc_" + self.bound_type + " = " + str(self.pert_perc) + "\n")
            f.write("pert_freq_" + self.bound_type + " = " + str(self.pert_freq) + "\n")

        # set "interior" solution
        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [1000.0, 1200.0],
            [0.6, 0.4],
        ])
        self.num_cells = 2
        self.sol = SolutionPhys(self.gas, self.num_cells, sol_prim_in=self.sol_prim_in)

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calc_subsonic_bc(self):

        # NOTE: need to set up new SystemSolver for each case because 
        # bound_cond is read from SystemSolver.param_dict, which is set at instantiation 
        self.press = 2e5
        with open(self.test_file, "a") as f:
            f.write('bound_cond_' + self.bound_type + ' = "subsonic"\n')
            f.write("press_" + self.bound_type + " = " + str(self.press) + "\n")
        solver = SystemSolver(self.test_dir)

        sol_bound = SolutionOutlet(self.gas, solver)
        sol_bound.calc_boundary_state(1e-6, 1, sol_prim=self.sol.sol_prim, sol_cons=self.sol.sol_cons)
        
        if self.output_mode:

            np.save(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_prim.npy"), sol_bound.sol_prim)
            np.save(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_cons.npy"), sol_bound.sol_cons)

        else:

            self.assertTrue(np.allclose(
                sol_bound.sol_prim,
                np.load(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_prim.npy"))
            ))
            self.assertTrue(np.allclose(
                sol_bound.sol_cons,
                np.load(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_cons.npy"))
            ))

    def test_calc_mean_flow_bc(self):

        self.press = 99830
        self.vel = 169
        self.rho = 328
        with open(self.test_file, "a") as f:
            f.write('bound_cond_' + self.bound_type + ' = "meanflow"\n')
            f.write("press_" + self.bound_type + " = " + str(self.press) + "\n")
            f.write("vel_" + self.bound_type + " = " + str(self.vel) + "\n")
            f.write("rho_" + self.bound_type + " = " + str(self.rho) + "\n")
        solver = SystemSolver(self.test_dir)

        sol_bound = SolutionOutlet(self.gas, solver)
        sol_bound.calc_boundary_state(0, 1, sol_prim=self.sol.sol_prim, sol_cons=self.sol.sol_cons)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_prim.npy"), sol_bound.sol_prim)
            np.save(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_cons.npy"), sol_bound.sol_cons)

        else:

            self.assertTrue(np.allclose(
                sol_bound.sol_prim,
                np.load(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_prim.npy"))
            ))
            self.assertTrue(np.allclose(
                sol_bound.sol_cons,
                np.load(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_cons.npy"))
            ))
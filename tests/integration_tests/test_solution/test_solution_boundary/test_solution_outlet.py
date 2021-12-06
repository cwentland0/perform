import unittest
import os

import numpy as np

from constants import CHEM_DICT_REACT, SOL_PRIM_IN_REACT, TEST_DIR, del_test_dir, gen_test_dir, get_output_mode
import perform.constants as constants
from perform.system_solver import SystemSolver
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_phys import SolutionPhys
from perform.solution.solution_boundary.solution_outlet import SolutionOutlet


class SolutionOutletMethodTests(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # set chemistry
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        # generate working directory
        gen_test_dir()

        # NOTE: press, vel, temp, and rho set later, depending on boundary condition
        self.bound_type = "outlet"
        self.mass_fracs = [0.4, 0.6]
        self.pert_type = "pressure"
        self.pert_perc = 0.025
        self.pert_freq = [125e3]

        # generate file with necessary input files
        self.test_file = os.path.join(TEST_DIR, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-7\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

            f.write("mass_fracs_" + self.bound_type + " = " + str(self.mass_fracs) + "\n")
            f.write("pert_type_" + self.bound_type + ' = "' + str(self.pert_type) + '"\n')
            f.write("pert_perc_" + self.bound_type + " = " + str(self.pert_perc) + "\n")
            f.write("pert_freq_" + self.bound_type + " = " + str(self.pert_freq) + "\n")

        # set "interior" solution
        self.num_cells = 2
        self.sol = SolutionPhys(self.gas, self.num_cells, sol_prim_in=SOL_PRIM_IN_REACT)

    def tearDown(self):

        del_test_dir()

    def test_calc_subsonic_bc(self):

        # NOTE: need to set up new SystemSolver for each case because
        # bound_cond is read from SystemSolver.param_dict, which is set at instantiation
        self.press = 2e5
        with open(self.test_file, "a") as f:
            f.write("bound_cond_" + self.bound_type + ' = "subsonic"\n')
            f.write("press_" + self.bound_type + " = " + str(self.press) + "\n")
        solver = SystemSolver(TEST_DIR)

        sol_bound = SolutionOutlet(self.gas, solver)
        sol_bound.calc_boundary_state(1e-6, 2, sol_prim=self.sol.sol_prim, sol_cons=self.sol.sol_cons)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_prim.npy"), sol_bound.sol_prim)
            np.save(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_cons.npy"), sol_bound.sol_cons)

        else:

            self.assertTrue(
                np.allclose(
                    sol_bound.sol_prim, np.load(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_prim.npy"))
                )
            )
            self.assertTrue(
                np.allclose(
                    sol_bound.sol_cons, np.load(os.path.join(self.output_dir, "bound_outlet_subsonic_sol_cons.npy"))
                )
            )

    def test_calc_mean_flow_bc(self):

        self.press = 898477.0
        self.vel = 1522.0
        self.rho = 2958.0
        with open(self.test_file, "a") as f:
            f.write("bound_cond_" + self.bound_type + ' = "meanflow"\n')
            f.write("press_" + self.bound_type + " = " + str(self.press) + "\n")
            f.write("vel_" + self.bound_type + " = " + str(self.vel) + "\n")
            f.write("rho_" + self.bound_type + " = " + str(self.rho) + "\n")
        solver = SystemSolver(TEST_DIR)

        sol_bound = SolutionOutlet(self.gas, solver)
        sol_bound.calc_boundary_state(0, 2, sol_prim=self.sol.sol_prim, sol_cons=self.sol.sol_cons)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_prim.npy"), sol_bound.sol_prim)
            np.save(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_cons.npy"), sol_bound.sol_cons)

        else:

            self.assertTrue(
                np.allclose(
                    sol_bound.sol_prim, np.load(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_prim.npy"))
                )
            )
            self.assertTrue(
                np.allclose(
                    sol_bound.sol_cons, np.load(os.path.join(self.output_dir, "bound_outlet_meanflow_sol_cons.npy"))
                )
            )

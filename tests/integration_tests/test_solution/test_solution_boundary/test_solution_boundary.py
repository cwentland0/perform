import unittest
import os

import numpy as np

from constants import CHEM_DICT_REACT, TEST_DIR, del_test_dir, gen_test_dir, get_output_mode
import perform.constants as constants
from perform.system_solver import SystemSolver
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_boundary.solution_boundary import SolutionBoundary


class SolutionBoundaryInitTestCase(unittest.TestCase):
    def setUp(self):
        # set chemistry
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        # generate working directory
        gen_test_dir()

        self.bound_type = "inlet"
        self.press = 1e6
        self.vel = 2.0
        self.temp = 300.0
        self.mass_fracs = [0.6, 0.4]
        self.rho = 11.8
        self.pert_type = "pressure"
        self.pert_perc = 0.025
        self.pert_freq = [125e3]

        # generate file with necessary input files
        self.test_file = os.path.join(TEST_DIR, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-8\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

            f.write("press_" + self.bound_type + " = " + str(self.press) + "\n")
            f.write("vel_" + self.bound_type + " = " + str(self.vel) + "\n")
            f.write("temp_" + self.bound_type + " = " + str(self.temp) + "\n")
            f.write("mass_fracs_" + self.bound_type + " = " + str(self.mass_fracs) + "\n")
            f.write("rho_" + self.bound_type + " = " + str(self.rho) + "\n")
            f.write("pert_type_" + self.bound_type + ' = "' + str(self.pert_type) + '"\n')
            f.write("pert_perc_" + self.bound_type + " = " + str(self.pert_perc) + "\n")
            f.write("pert_freq_" + self.bound_type + " = " + str(self.pert_freq) + "\n")

        # set SystemSolver
        self.solver = SystemSolver(TEST_DIR)

    def tearDown(self):

        del_test_dir()

    def test_solution_boundary_init(self):

        sol = SolutionBoundary(self.gas, self.solver, "inlet")

        self.assertEqual(sol.press, self.press)
        self.assertEqual(sol.vel, self.vel)
        self.assertEqual(sol.temp, self.temp)
        self.assertTrue(np.array_equal(sol.mass_fracs, self.mass_fracs))
        self.assertEqual(sol.rho, self.rho)
        self.assertEqual(sol.pert_type, self.pert_type)
        self.assertEqual(sol.pert_perc, self.pert_perc)
        self.assertTrue(np.array_equal(sol.pert_freq, self.pert_freq))


class SolutionBoundaryMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # set chemistry
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        # generate working directory
        gen_test_dir()

        self.bound_type = "inlet"
        self.press = 1e6
        self.vel = 2.0
        self.temp = 300.0
        self.mass_fracs = [0.6, 0.4]
        self.rho = 11.8
        self.pert_type = "pressure"
        self.pert_perc = 0.025
        self.pert_freq = [125e3]

        # generate file with necessary input files
        self.test_file = os.path.join(TEST_DIR, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-8\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

            f.write("press_" + self.bound_type + " = " + str(self.press) + "\n")
            f.write("vel_" + self.bound_type + " = " + str(self.vel) + "\n")
            f.write("temp_" + self.bound_type + " = " + str(self.temp) + "\n")
            f.write("mass_fracs_" + self.bound_type + " = " + str(self.mass_fracs) + "\n")
            f.write("rho_" + self.bound_type + " = " + str(self.rho) + "\n")
            f.write("pert_type_" + self.bound_type + ' = "' + str(self.pert_type) + '"\n')
            f.write("pert_perc_" + self.bound_type + " = " + str(self.pert_perc) + "\n")
            f.write("pert_freq_" + self.bound_type + " = " + str(self.pert_freq) + "\n")

        # set SystemSolver
        self.solver = SystemSolver(TEST_DIR)

        # set solution
        self.sol = SolutionBoundary(self.gas, self.solver, "inlet")

    def tearDown(self):

        del_test_dir()

    def test_calc_pert(self):

        pert = self.sol.calc_pert(1e-6)

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "bound_pert.npy"), np.array([pert]))
        else:
            self.assertTrue(np.allclose(np.array([pert]), np.load(os.path.join(self.output_dir, "bound_pert.npy"))))

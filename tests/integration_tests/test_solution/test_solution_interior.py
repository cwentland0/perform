import unittest
import os
import shutil

import numpy as np

from constants import CHEM_DICT_REACT
import perform.constants as constants
from perform.system_solver import SystemSolver
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.time_integrator.implicit_integrator import BDF
from perform.solution.solution_interior import SolutionInterior


class SolutionIntInitTestCase(unittest.TestCase):
    def setUp(self):

        # set chemistry
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        # set time integrator
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "bdf"
        self.param_dict["time_order"] = 2
        self.time_int = BDF(self.param_dict)

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # generate file with necessary input files
        self.test_file = os.path.join(self.test_dir, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-8\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

        # set SystemSolver and time integrator
        self.solver = SystemSolver(self.test_dir)
        self.time_int = BDF(self.param_dict)

        # set primitive state
        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [1000.0, 1200.0],
            [0.6, 0.4],
        ])

        self.num_cells = 2
        self.num_reactions = 1

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_solution_int_init(self):

        sol = SolutionInterior(self.gas, self.sol_prim_in, self.solver, self.num_cells, self.num_reactions, self.time_int)
        self.assertTrue(np.array_equal(sol.sol_prim, self.sol_prim_in))
        self.assertTrue(np.allclose(
            sol.sol_cons,
            np.array([
                [2.56420677,  0.213683897],
                [5.12841354,  0.213683897],
                [-1.95673648e7, -1.72547963e6],
                [1.53852406,  8.54735590e-02],
            ])
        ))
    
        # TODO: a LOT of checking of other variables


class SolutionIntMethodsTestCase(unittest.TestCase):
    def setUp(self):

        # set chemistry
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        # set time integrator
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "bdf"
        self.param_dict["time_order"] = 2
        self.time_int = BDF(self.param_dict)

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # generate file with necessary input files
        self.test_file = os.path.join(self.test_dir, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-8\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

        # set SystemSolver and time integrator
        self.solver = SystemSolver(self.test_dir)
        self.time_int = BDF(self.param_dict)

        # set primitive state
        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [1000.0, 1200.0],
            [0.6, 0.4],
        ])

        self.num_cells = 2
        self.num_reactions = 1

        self.sol = SolutionInterior(self.gas, self.sol_prim_in, self.solver, self.num_cells, self.num_reactions, self.time_int)

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calc_sol_jacob(self):

        sol_jacob = self.sol.calc_sol_jacob(inverse=False)
        sol_jacob_inv = self.sol.calc_sol_jacob(inverse=True)

        pass
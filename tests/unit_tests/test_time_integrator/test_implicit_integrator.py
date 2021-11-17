import unittest
import shutil
import os

import numpy as np

from perform.time_integrator.implicit_integrator import ImplicitIntegrator, BDF
from perform.system_solver import SystemSolver
import perform.constants as constants


class ImplicitTimeIntInitTestCase(unittest.TestCase):
    def setUp(self):

        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "bdf"
        self.param_dict["time_order"] = 2
        self.param_dict["subiter_max"] = 10
        self.param_dict["res_tol"] = 1e-9
        self.param_dict["dtau"] = 1e-4
        self.param_dict["cfl"] = 2.0
        self.param_dict["vnn"] = 10.0

    def test_implicit_time_int_init(self):

        time_int = ImplicitIntegrator(self.param_dict)

        self.assertEqual(time_int.time_type, "implicit")
        self.assertTrue(time_int.dual_time)
        self.assertFalse(time_int.adapt_dtau)
        self.assertEqual(time_int.subiter_max, self.param_dict["subiter_max"])
        self.assertEqual(time_int.res_tol, self.param_dict["res_tol"])
        self.assertEqual(time_int.cold_start_iter, 1)
        self.assertEqual(time_int.cfl, self.param_dict["cfl"])
        self.assertEqual(time_int.vnn, self.param_dict["vnn"])

class BDFTestCase(unittest.TestCase):

    def setUp(self):
        
        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "bdf"
        self.param_dict["time_order"] = 2

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

        # set solution and RHS variables
        self.sol_hist = [
            np.array([[2.0, 5.0], [3.0, 6.0], [4.0, 7.0]]),
            np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ]
        self.rhs = np.array([[1e7, 4e7], [2e7, 5e7], [3e7, 6e7]])

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calc_residual(self):

        residual = self.time_int.calc_residual(self.sol_hist, self.rhs, self.solver)
        self.assertTrue(np.allclose(residual, np.array([[0.0, 3e7], [1e7, 4e7], [2e7, 5e7]])))

import unittest
import os
import shutil

import numpy as np

from constants import CHEM_DICT_REACT
import perform.constants as constants
from perform.system_solver import SystemSolver
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.time_integrator.implicit_integrator import BDF
from perform.solution.solution_domain import SolutionDomain


class SolutionDomainInitTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # generate mesh file
        self.mesh_file = os.path.join(self.test_dir, "mesh.inp")
        with open(self.mesh_file, "w") as f:
            f.write('x_left = 0.0\n')
            f.write('x_right = 2e-5\n')
            f.write('num_cells = 2\n')

        # generate chemistry file
        self.chem_file = os.path.join(self.test_dir, "chem.inp")
        with open(self.chem_file, "w") as f:
            for key, item in CHEM_DICT_REACT.items():
                if isinstance(item, str):
                    f.write(key + " = \"" + str(item) + "\"\n")
                else:
                    f.write(key + " = " + str(item) + "\n")

        # generate initial condition file
        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [1000.0, 1200.0],
            [0.6, 0.4],
        ])
        np.save(os.path.join(self.test_dir, "test_init_file.npy"), self.sol_prim_in)

        # generate solver input files
        self.inp_file = os.path.join(self.test_dir, constants.PARAM_INPUTS)
        with open(self.inp_file, "w") as f:
            f.write('chem_file = "' + self.chem_file + '"\n')
            f.write('mesh_file = "' + self.mesh_file + '"\n')
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-7\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")
            f.write("invisc_flux_scheme")
            f.write("visc_flux_scheme")
            f.write("space_order")
            f.write("grad_limiter")
            f.write("bound_cond_inlet")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")
            f.write("")

        # set SystemSolver and time integrator
        self.solver = SystemSolver(self.test_dir)


    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_solution_domain_init(self):

        pass
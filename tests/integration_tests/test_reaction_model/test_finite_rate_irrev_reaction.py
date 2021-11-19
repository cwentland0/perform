import unittest
import os
import shutil

import numpy as np

from constants import CHEM_DICT_REACT
import perform.constants as constants
from perform.system_solver import SystemSolver
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.reaction_model.finite_rate_irrev_reaction import FiniteRateIrrevReaction
from perform.time_integrator.implicit_integrator import BDF
from perform.solution.solution_interior import SolutionInterior

class FiniteRateIrrevReactionInitTestCase(unittest.TestCase):
    def setUp(self):
        
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)
        
    def test_finite_rate_irrev_reaction_init(self):

        reaction_model = FiniteRateIrrevReaction(self.gas, self.chem_dict)

        self.assertTrue(np.array_equal(reaction_model.nu, np.array(CHEM_DICT_REACT["nu"])))
        self.assertTrue(np.array_equal(reaction_model.nu_arr, np.array(CHEM_DICT_REACT["nu_arr"])))
        self.assertTrue(np.array_equal(reaction_model.pre_exp_fact, np.array(CHEM_DICT_REACT["pre_exp_fact"])))
        self.assertTrue(np.array_equal(reaction_model.temp_exp, np.array(CHEM_DICT_REACT["temp_exp"])))
        self.assertTrue(np.array_equal(reaction_model.act_energy, np.array(CHEM_DICT_REACT["act_energy"])))


class FiniteRateIrrevReactionMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(os.environ["PERFORM_TEST_OUTPUT_MODE"])
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        # setup gas and reaction model
        self.chem_dict = CHEM_DICT_REACT
        self.gas = CaloricallyPerfectGas(self.chem_dict)
        self.reaction_model = FiniteRateIrrevReaction(self.gas, self.chem_dict)

        self.dt = 1e-7

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

        # setup solution
        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [1000.0, 1200.0],
            [0.6, 0.4],
        ])
        self.num_cells = 2
        self.num_reactions = 1
        self.sol = SolutionInterior(
            self.gas,
            self.sol_prim_in,
            self.solver,
            self.num_cells,
            self.num_reactions,
            self.time_int
        )

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calc_reaction(self):

        self.sol.update_density_enthalpy_derivs()
        source, wf, heat_release = self.reaction_model.calc_reaction(self.sol, self.dt)


        if self.output_mode:

            np.save(os.path.join(self.output_dir, "frirrev_source.npy"), source)
            np.save(os.path.join(self.output_dir, "frirrev_wf.npy"), wf)
            np.save(os.path.join(self.output_dir, "frirrev_heat_release.npy"), heat_release)

        else:

            self.assertTrue(np.allclose(
                source,
                np.load(os.path.join(self.output_dir, "frirrev_source.npy"))
            ))
            self.assertTrue(np.allclose(
                wf,
                np.load(os.path.join(self.output_dir, "frirrev_wf.npy"))
            ))
            self.assertTrue(np.allclose(
                heat_release,
                np.load(os.path.join(self.output_dir, "frirrev_heat_release.npy"))
            ))

    def test_calc_jacob(self):

        # need to compute density derivatives and wf beforehand
        self.sol.update_density_enthalpy_derivs()
        _, self.sol.wf = self.reaction_model.calc_source(self.sol, self.dt)
        jacob_prim = self.reaction_model.calc_jacob(self.sol, True)

        # TODO: test conservative Jacobian when it's implemented
        
        if self.output_mode:

            np.save(os.path.join(self.output_dir, "frirrev_jacob_prim.npy"), jacob_prim)

        else:

            self.assertTrue(np.allclose(
                jacob_prim,
                np.load(os.path.join(self.output_dir, "frirrev_jacob_prim.npy"))
            ))

    
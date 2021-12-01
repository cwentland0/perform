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

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

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
            f.write("dt = 1e-7\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")

        # set SystemSolver
        self.solver = SystemSolver(self.test_dir)

        # set primitive state
        self.sol_prim_in = np.array(
            [
                [1e6, 1e5],
                [2.0, 1.0],
                [1000.0, 1200.0],
                [0.6, 0.4],
            ]
        )

        self.num_cells = 2
        self.num_reactions = 1

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_solution_int_init(self):

        sol = SolutionInterior(
            self.gas, self.sol_prim_in, self.solver, self.num_cells, self.num_reactions, self.time_int
        )

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_int_init_sol_cons.npy"), sol.sol_cons)

        else:

            self.assertTrue(np.array_equal(sol.sol_prim, self.sol_prim_in))
            self.assertTrue(
                np.allclose(sol.sol_cons, np.load(os.path.join(self.output_dir, "sol_int_init_sol_cons.npy")))
            )

            # TODO: a LOT of checking of other variables


class SolutionIntMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

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
            f.write('init_file = "test_init_file.npy" \n')
            f.write("dt = 1e-8 \n")
            f.write('time_scheme = "bdf" \n')
            f.write("num_steps = 10 \n")
            f.write("out_interval = 2 \n")
            f.write("out_itmdt_interval = 5 \n")
            f.write("prim_out = True \n")
            f.write("cons_out = True \n")
            f.write("source_out = True \n")
            f.write("hr_out = True \n")
            f.write("rhs_out = True \n")

        # set SystemSolver
        self.solver = SystemSolver(self.test_dir)

        # set primitive state
        self.sol_prim_in = np.array(
            [
                [1e6, 1e5],
                [2.0, 1.0],
                [1000.0, 1200.0],
                [0.6, 0.4],
            ]
        )

        self.num_cells = 2
        self.num_reactions = 1

        self.sol = SolutionInterior(
            self.gas, self.sol_prim_in, self.solver, self.num_cells, self.num_reactions, self.time_int
        )

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calc_sol_jacob(self):

        sol_jacob = self.sol.calc_sol_jacob(inverse=False)
        sol_jacob_inv = self.sol.calc_sol_jacob(inverse=True)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_int_sol_jacob.npy"), sol_jacob)
            np.save(os.path.join(self.output_dir, "sol_int_sol_jacob_inv.npy"), sol_jacob_inv)

        else:

            self.assertTrue(np.allclose(sol_jacob, np.load(os.path.join(self.output_dir, "sol_int_sol_jacob.npy"))))
            self.assertTrue(
                np.allclose(sol_jacob_inv, np.load(os.path.join(self.output_dir, "sol_int_sol_jacob_inv.npy")))
            )

    def test_update_snapshots(self):

        # update the snapshot matrix
        for self.solver.iter in range(1, self.solver.num_steps + 1):
            if (self.solver.iter % self.solver.out_interval) == 0:
                self.sol.update_snapshots(self.solver)

        self.assertTrue(np.array_equal(self.sol.prim_snap, np.repeat(self.sol.sol_prim[:, :, None], 6, axis=2)))
        self.assertTrue(np.array_equal(self.sol.cons_snap, np.repeat(self.sol.sol_cons[:, :, None], 6, axis=2)))
        self.assertTrue(
            np.array_equal(self.sol.reaction_source_snap, np.repeat(self.sol.reaction_source[:, :, None], 5, axis=2))
        )
        self.assertTrue(
            np.array_equal(self.sol.heat_release_snap, np.repeat(self.sol.heat_release[:, None], 5, axis=1))
        )
        self.assertTrue(np.array_equal(self.sol.rhs_snap, np.repeat(self.sol.rhs[:, :, None], 5, axis=2)))

    def test_snapshot_output(self):

        for self.solver.iter in range(1, self.solver.num_steps + 1):

            # update the snapshot matrix
            if (self.solver.iter % self.solver.out_interval) == 0:
                self.sol.update_snapshots(self.solver)

            # write and check intermediate results
            if ((self.solver.iter % self.solver.out_itmdt_interval) == 0) and (
                self.solver.iter != self.solver.num_steps
            ):
                self.sol.write_snapshots(self.solver, intermediate=True, failed=False)

                sol_prim_itmdt = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "sol_prim_" + self.solver.sim_type + "_ITMDT.npy")
                )
                sol_cons_itmdt = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "sol_cons_" + self.solver.sim_type + "_ITMDT.npy")
                )
                source_itmdt = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "source_" + self.solver.sim_type + "_ITMDT.npy")
                )
                heat_release_itmdt = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "heat_release_" + self.solver.sim_type + "_ITMDT.npy")
                )
                rhs_itmdt = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "rhs_" + self.solver.sim_type + "_ITMDT.npy")
                )
                self.assertTrue(np.array_equal(sol_prim_itmdt, np.repeat(self.sol.sol_prim[:, :, None], 3, axis=2)))
                self.assertTrue(np.array_equal(sol_cons_itmdt, np.repeat(self.sol.sol_cons[:, :, None], 3, axis=2)))
                self.assertTrue(
                    np.array_equal(source_itmdt, np.repeat(self.sol.reaction_source[:, :, None], 2, axis=2))
                )
                self.assertTrue(
                    np.array_equal(heat_release_itmdt, np.repeat(self.sol.heat_release[:, None], 2, axis=1))
                )
                self.assertTrue(np.array_equal(rhs_itmdt, np.repeat(self.sol.rhs[:, :, None], 2, axis=2)))

            # write and check "failed" snapshots
            if self.solver.iter == 7:
                self.sol.write_snapshots(self.solver, intermediate=False, failed=True)

                sol_prim_failed = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "sol_prim_" + self.solver.sim_type + "_FAILED.npy")
                )
                sol_cons_failed = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "sol_cons_" + self.solver.sim_type + "_FAILED.npy")
                )
                source_failed = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "source_" + self.solver.sim_type + "_FAILED.npy")
                )
                heat_release_failed = np.load(
                    os.path.join(
                        self.solver.unsteady_output_dir, "heat_release_" + self.solver.sim_type + "_FAILED.npy"
                    )
                )
                rhs_failed = np.load(
                    os.path.join(self.solver.unsteady_output_dir, "rhs_" + self.solver.sim_type + "_FAILED.npy")
                )
                self.assertTrue(np.array_equal(sol_prim_failed, np.repeat(self.sol.sol_prim[:, :, None], 4, axis=2)))
                self.assertTrue(np.array_equal(sol_cons_failed, np.repeat(self.sol.sol_cons[:, :, None], 4, axis=2)))
                self.assertTrue(
                    np.array_equal(source_failed, np.repeat(self.sol.reaction_source[:, :, None], 3, axis=2))
                )
                self.assertTrue(
                    np.array_equal(heat_release_failed, np.repeat(self.sol.heat_release[:, None], 3, axis=1))
                )
                self.assertTrue(np.array_equal(rhs_failed, np.repeat(self.sol.rhs[:, :, None], 3, axis=2)))

        # delete intermediate results and check that they deleted properly
        self.sol.delete_itmdt_snapshots(self.solver)
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.unsteady_output_dir, "sol_prim_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.unsteady_output_dir, "sol_cons_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.unsteady_output_dir, "source_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.unsteady_output_dir, "heat_release_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )
        self.assertFalse(
            os.path.isfile(os.path.join(self.solver.unsteady_output_dir, "rhs_" + self.solver.sim_type + "_ITMDT.npy"))
        )

        # write final snapshots
        self.sol.write_snapshots(self.solver, intermediate=False, failed=False)
        sol_prim_final = np.load(
            os.path.join(self.solver.unsteady_output_dir, "sol_prim_" + self.solver.sim_type + ".npy")
        )
        sol_cons_final = np.load(
            os.path.join(self.solver.unsteady_output_dir, "sol_cons_" + self.solver.sim_type + ".npy")
        )
        source_final = np.load(os.path.join(self.solver.unsteady_output_dir, "source_" + self.solver.sim_type + ".npy"))
        heat_release_final = np.load(
            os.path.join(self.solver.unsteady_output_dir, "heat_release_" + self.solver.sim_type + ".npy")
        )
        rhs_final = np.load(os.path.join(self.solver.unsteady_output_dir, "rhs_" + self.solver.sim_type + ".npy"))
        self.assertTrue(np.array_equal(sol_prim_final, np.repeat(self.sol.sol_prim[:, :, None], 6, axis=2)))
        self.assertTrue(np.array_equal(sol_cons_final, np.repeat(self.sol.sol_cons[:, :, None], 6, axis=2)))
        self.assertTrue(np.array_equal(source_final, np.repeat(self.sol.reaction_source[:, :, None], 5, axis=2)))
        self.assertTrue(np.array_equal(heat_release_final, np.repeat(self.sol.heat_release[:, None], 5, axis=1)))
        self.assertTrue(np.array_equal(rhs_final, np.repeat(self.sol.rhs[:, :, None], 5, axis=2)))

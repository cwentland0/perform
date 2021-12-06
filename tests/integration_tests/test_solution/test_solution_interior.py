import unittest
import os

import numpy as np

from constants import (
    del_test_dir,
    gen_test_dir,
    get_output_mode,
    solution_domain_setup,
    CHEM_DICT_REACT,
    SOL_PRIM_IN_REACT,
    TEST_DIR,
)
from perform.constants import REAL_TYPE
from perform.system_solver import SystemSolver
from perform.input_funcs import read_restart_file
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.time_integrator.implicit_integrator import BDF
from perform.solution.solution_interior import SolutionInterior


class SolutionIntInitTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

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
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver
        self.solver = SystemSolver(TEST_DIR)

        self.num_cells = 2
        self.num_reactions = 1

    def tearDown(self):

        del_test_dir()

    def test_solution_int_init(self):

        sol = SolutionInterior(
            self.gas, SOL_PRIM_IN_REACT, self.solver, self.num_cells, self.num_reactions, self.time_int
        )

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_int_init_sol_cons.npy"), sol.sol_cons)

        else:

            self.assertTrue(np.array_equal(sol.sol_prim, SOL_PRIM_IN_REACT))
            self.assertTrue(
                np.allclose(sol.sol_cons, np.load(os.path.join(self.output_dir, "sol_int_init_sol_cons.npy")))
            )

            # TODO: a LOT of checking of other variables


class SolutionIntMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

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
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver
        self.solver = SystemSolver(TEST_DIR)

        self.num_cells = 2
        self.num_reactions = 1

        self.sol = SolutionInterior(
            self.gas, SOL_PRIM_IN_REACT, self.solver, self.num_cells, self.num_reactions, self.time_int
        )

    def tearDown(self):

        del_test_dir()

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

    def test_write_restart_file(self):

        sol_cons = self.sol.sol_cons

        self.solver.sol_time = 1e-4
        self.solver.iter = 2
        self.solver.restart_iter = 4
        self.sol.write_restart_file(self.solver)

        self.assertEqual(self.solver.restart_iter, 5)

        # check restart files
        restart_data = np.load(os.path.join(self.solver.restart_output_dir, "restart_file_4.npz"))

        self.assertTrue(
            np.array_equal(
                restart_data["sol_prim"],
                np.repeat(SOL_PRIM_IN_REACT[:, :, None], 2, axis=-1),
            )
        )
        self.assertTrue(
            np.array_equal(
                restart_data["sol_cons"],
                np.repeat(sol_cons[:, :, None], 2, axis=-1),
            )
        )
        self.assertEqual(float(restart_data["sol_time"]), 1e-4)

        # check iteration files
        restart_iter = int(np.loadtxt(os.path.join(self.solver.restart_output_dir, "restart_iter.dat")))
        self.assertEqual(restart_iter, 4)

    def test_read_restart_file(self):

        self.solver.sol_time = 1e-4
        self.solver.iter = 2
        self.solver.restart_iter = 4
        self.sol.write_restart_file(self.solver)

        sol_time, sol_prim, restart_iter = read_restart_file(self.solver)
        self.assertEqual(sol_time, 1e-4)
        self.assertEqual(restart_iter, 5)  # 1 is added to avoid overwriting
        self.assertTrue(
            np.array_equal(
                sol_prim,
                np.repeat(SOL_PRIM_IN_REACT[:, :, None], 2, axis=-1),
            )
        )

    def test_calc_d_sol_norms(self):

        self.solver.iter = 3
        self.sol.d_sol_norm_hist = np.zeros((self.solver.num_steps, 2), dtype=REAL_TYPE)
        self.sol.sol_hist_prim[0] = self.sol.sol_prim * 2.0

        self.sol.calc_d_sol_norms(self.solver, "implicit")

        self.assertAlmostEqual(self.sol.d_sol_norm_hist[2, 0], 3.46573790883)
        self.assertAlmostEqual(self.sol.d_sol_norm_hist[2, 1], 3.45416666667)

    def test_calc_res_norms(self):

        self.solver.iter = 3
        self.sol.res = self.sol.sol_prim.copy()

        self.sol.calc_res_norms(self.solver, 0)

        self.assertAlmostEqual(self.sol.res_norm_hist[2, 0], 3.46573790883)
        self.assertAlmostEqual(self.sol.res_norm_hist[2, 1], 3.45416666667)

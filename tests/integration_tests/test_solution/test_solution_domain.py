import unittest
import os
import shutil

import numpy as np

from constants import solution_domain_setup
from perform.system_solver import SystemSolver
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

        # generate input text files
        solution_domain_setup(self.test_dir)

        # generate initial condition file
        self.sol_prim_in = np.array(
            [
                [1e6, 9e5],
                [2.0, 1.0],
                [1000.0, 1200.0],
                [0.6, 0.4],
            ]
        )
        np.save(os.path.join(self.test_dir, "test_init_file.npy"), self.sol_prim_in)

        # set SystemSolver
        self.solver = SystemSolver(self.test_dir)

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_solution_domain_init(self):

        sol = SolutionDomain(self.solver)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_domain_int_init_sol_cons.npy"), sol.sol_int.sol_cons)

        else:

            self.assertTrue(np.array_equal(sol.sol_int.sol_prim, self.sol_prim_in))
            self.assertTrue(
                np.allclose(
                    sol.sol_int.sol_cons, np.load(os.path.join(self.output_dir, "sol_domain_int_init_sol_cons.npy"))
                )
            )

            # TODO: a LOT of checking of other variables


class SolutionDomainMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # generate input text files
        solution_domain_setup(self.test_dir)

        # generate initial condition file
        self.sol_prim_in = np.array(
            [
                [1e6, 9e5],
                [2.0, 1.0],
                [1000.0, 1200.0],
                [0.6, 0.4],
            ]
        )
        np.save(os.path.join(self.test_dir, "test_init_file.npy"), self.sol_prim_in)

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(self.test_dir)
        self.sol_domain = SolutionDomain(self.solver)

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_calc_rhs(self):

        self.sol_domain.calc_rhs(self.solver)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_domain_rhs.npy"), self.sol_domain.sol_int.rhs)

        else:

            self.assertTrue(
                np.allclose(self.sol_domain.sol_int.rhs, np.load(os.path.join(self.output_dir, "sol_domain_rhs.npy")))
            )

    def test_calc_rhs_jacob(self):

        self.sol_domain.calc_rhs(self.solver)
        rhs_jacob_center, rhs_jacob_left, rhs_jacob_right = self.sol_domain.calc_rhs_jacob(self.solver)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_domain_rhs_jacob_center.npy"), rhs_jacob_center)
            np.save(os.path.join(self.output_dir, "sol_domain_rhs_jacob_left.npy"), rhs_jacob_left)
            np.save(os.path.join(self.output_dir, "sol_domain_rhs_jacob_right.npy"), rhs_jacob_right)

        else:

            self.assertTrue(
                np.allclose(rhs_jacob_center, np.load(os.path.join(self.output_dir, "sol_domain_rhs_jacob_center.npy")))
            )
            self.assertTrue(
                np.allclose(rhs_jacob_left, np.load(os.path.join(self.output_dir, "sol_domain_rhs_jacob_left.npy")))
            )
            self.assertTrue(
                np.allclose(rhs_jacob_right, np.load(os.path.join(self.output_dir, "sol_domain_rhs_jacob_right.npy")))
            )

    def test_calc_res_jacob(self):

        self.sol_domain.calc_rhs(self.solver)
        res_jacob = self.sol_domain.calc_res_jacob(self.solver).todense()

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_domain_res_jacob.npy"), res_jacob)

        else:

            self.assertTrue(np.allclose(res_jacob, np.load(os.path.join(self.output_dir, "sol_domain_res_jacob.npy"))))

    def test_advance_subiter(self):

        self.sol_domain.time_integrator.subiter = 0
        self.sol_domain.advance_subiter(self.solver)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_domain_subiter_sol_prim.npy"), self.sol_domain.sol_int.sol_prim)
            np.save(os.path.join(self.output_dir, "sol_domain_subiter_sol_cons.npy"), self.sol_domain.sol_int.sol_cons)
            np.save(os.path.join(self.output_dir, "sol_domain_subiter_res.npy"), self.sol_domain.sol_int.res)

        else:

            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_prim,
                    np.load(os.path.join(self.output_dir, "sol_domain_subiter_sol_prim.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_cons,
                    np.load(os.path.join(self.output_dir, "sol_domain_subiter_sol_cons.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.res, np.load(os.path.join(self.output_dir, "sol_domain_subiter_res.npy"))
                )
            )

    def test_advance_iter(self):

        self.sol_domain.advance_iter(self.solver)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_domain_iter_sol_prim.npy"), self.sol_domain.sol_int.sol_prim)
            np.save(os.path.join(self.output_dir, "sol_domain_iter_sol_cons.npy"), self.sol_domain.sol_int.sol_cons)
            np.save(os.path.join(self.output_dir, "sol_domain_iter_res.npy"), self.sol_domain.sol_int.res)

        else:

            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_prim,
                    np.load(os.path.join(self.output_dir, "sol_domain_iter_sol_prim.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_cons,
                    np.load(os.path.join(self.output_dir, "sol_domain_iter_sol_cons.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.res, np.load(os.path.join(self.output_dir, "sol_domain_iter_res.npy"))
                )
            )

import unittest
import os
import shutil

import numpy as np

from constants import solution_domain_setup
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.limiter.venkat_limiter import VenkatLimiter
from perform.limiter.barth_jesp_limiter import BarthJespCellLimiter, BarthJespFaceLimiter


class LimiterMethodsTestCase(unittest.TestCase):
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

        # calculate raw gradients
        self.sol_domain.calc_ghost_cells(self.solver)
        self.sol_domain.fill_sol_full()
        self.grad = (0.5 / self.sol_domain.mesh.dx) * (
            self.sol_domain.sol_prim_full[:, self.sol_domain.grad_idxs + 1]
            - self.sol_domain.sol_prim_full[:, self.sol_domain.grad_idxs - 1]
        )

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_venkat_limiter(self):

        limiter = VenkatLimiter()
        phi = limiter.calc_limiter(self.sol_domain, self.sol_domain.sol_prim_full, self.grad)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "venkat_limiter_phi.npy"), phi)

        else:

            self.assertTrue(np.allclose(phi, np.load(os.path.join(self.output_dir, "venkat_limiter_phi.npy"))))

    def test_barth_cell_limiter(self):

        limiter = BarthJespCellLimiter()
        phi = limiter.calc_limiter(self.sol_domain, self.sol_domain.sol_prim_full, self.grad)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "barth_cell_limiter_phi.npy"), phi)

        else:

            self.assertTrue(np.allclose(phi, np.load(os.path.join(self.output_dir, "barth_cell_limiter_phi.npy"))))

    def test_barth_face_limiter(self):

        limiter = BarthJespFaceLimiter()
        phi = limiter.calc_limiter(self.sol_domain, self.sol_domain.sol_prim_full, self.grad)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "barth_face_limiter_phi.npy"), phi)

        else:

            self.assertTrue(np.allclose(phi, np.load(os.path.join(self.output_dir, "barth_face_limiter_phi.npy"))))

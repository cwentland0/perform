import unittest
import os

import numpy as np

from constants import TEST_DIR, del_test_dir, gen_test_dir, get_output_mode, solution_domain_setup
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.limiter.venkat_limiter import VenkatLimiter
from perform.limiter.barth_jesp_limiter import BarthJespCellLimiter, BarthJespFaceLimiter


class LimiterMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

        # calculate raw gradients
        self.sol_domain.calc_ghost_cells(self.solver)
        self.sol_domain.fill_sol_full()
        self.grad = (0.5 / self.sol_domain.mesh.dx) * (
            self.sol_domain.sol_prim_full[:, self.sol_domain.grad_idxs + 1]
            - self.sol_domain.sol_prim_full[:, self.sol_domain.grad_idxs - 1]
        )

    def tearDown(self):

        del_test_dir()

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

import unittest

import numpy as np

from constants import TEST_DIR, del_test_dir, gen_test_dir, solution_domain_setup
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.visualization_group import VisualizationGroup


class VisGroupInitTestCase(unittest.TestCase):
    def setUp(self):

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

    def tearDown(self):

        del_test_dir()

    def test_vis_group_init(self):

        vis_group = VisualizationGroup(self.sol_domain, self.solver)

        self.assertEqual(vis_group.num_vis_plots, 2)

        # check what should be a field plot
        plot = vis_group.vis_list[0]
        self.assertEqual(plot.vis_type, "field")
        self.assertEqual(plot.num_imgs, 3)
        self.assertEqual(plot.num_subplots, 4)
        self.assertEqual(plot.num_rows, 2)
        self.assertEqual(plot.num_cols, 2)

        # check what should be a probe plot
        plot = vis_group.vis_list[1]
        self.assertEqual(plot.vis_type, "probe")
        self.assertEqual(plot.num_subplots, 4)
        self.assertEqual(plot.num_rows, 2)
        self.assertEqual(plot.num_cols, 2)


class VisGroupMethodsTestCase(unittest.TestCase):
    def setUp(self):

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

        self.vis_group = VisualizationGroup(self.sol_domain, self.solver)

    def tearDown(self):

        del_test_dir()

    def test_draw_plots(self):

        vis_vals = [
            self.sol_domain.sol_int.sol_prim[2, :],
            self.sol_domain.sol_int.sol_cons[0, :],
            self.sol_domain.sol_int.sol_prim[0, :],
            self.sol_domain.sol_int.sol_prim[3, :],
        ]

        # loop outer iterations
        for self.solver.iter in range(1, self.solver.num_steps + 1):

            self.sol_domain.update_probes(self.solver)
            self.vis_group.draw_plots(self.sol_domain, self.solver)

            if (self.solver.iter % self.vis_group.vis_interval) != 0:
                continue

            # check field plot
            plot = self.vis_group.vis_list[0]
            for ax_idx in range(plot.num_subplots):
                ax = plot.ax.ravel()[ax_idx]
                self.assertTrue(np.array_equal(vis_vals[ax_idx], ax.lines[0].get_xydata()[:, 1]))

            # check probe plot
            plot = self.vis_group.vis_list[1]
            for ax_idx in range(plot.num_subplots):
                ax = plot.ax.ravel()[ax_idx]
                if ax_idx == 0:
                    var_idx = 1
                elif ax_idx == 1:
                    var_idx = 3
                elif ax_idx == 2:
                    var_idx = 0
                elif ax_idx == 3:
                    var_idx = 2

                self.assertTrue(
                    np.array_equal(
                        self.sol_domain.probe_vals[1, var_idx, : self.solver.iter], ax.lines[0].get_xydata()[:, 1]
                    )
                )

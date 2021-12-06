import unittest

import numpy as np

from constants import TEST_DIR, del_test_dir, gen_test_dir, solution_domain_setup
from perform.input_funcs import catch_list
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.field_plot import FieldPlot


class FieldPlotInitTestCase(unittest.TestCase):
    def setUp(self):

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

        # get some plotting variables
        param_dict = self.solver.param_dict
        self.vis_vars = catch_list(param_dict, "vis_var_0", [None])
        self.vis_x_bounds = catch_list(param_dict, "vis_x_bounds_0", [[None, None]], len_highest=len(self.vis_vars))
        self.vis_y_bounds = catch_list(param_dict, "vis_y_bounds_0", [[None, None]], len_highest=len(self.vis_vars))

    def tearDown(self):

        del_test_dir()

    def test_field_plot_init(self):

        plot = FieldPlot(
            self.solver.image_output_dir,
            0,
            1,
            self.solver.num_steps,
            self.solver.sim_type,
            self.vis_vars,
            self.vis_x_bounds,
            self.vis_y_bounds,
            self.sol_domain.gas_model.species_names,
        )

        self.assertEqual(plot.vis_type, "field")
        self.assertEqual(plot.num_imgs, 10)
        self.assertEqual(plot.num_subplots, 4)
        self.assertEqual(plot.num_rows, 2)
        self.assertEqual(plot.num_cols, 2)

        for bound_idx, bound in enumerate(self.vis_x_bounds):
            self.assertTrue(np.array_equal(bound, plot.vis_x_bounds[bound_idx]))
            self.assertTrue(np.array_equal(self.vis_y_bounds[bound_idx], plot.vis_y_bounds[bound_idx]))


class FieldPlotMethodsTestCase(unittest.TestCase):
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

    def test_field_plot_plot(self):

        # set plotting variables
        vis_vars = [
            "pressure",
            "velocity",
            "temperature",
            "species-0",
            "density",
            "momentum",
            "energy",
            "density-species-0",
            "source-0",
            "heat-release",
        ]
        vis_vals = [
            self.sol_domain.sol_int.sol_prim[0, :],
            self.sol_domain.sol_int.sol_prim[1, :],
            self.sol_domain.sol_int.sol_prim[2, :],
            self.sol_domain.sol_int.sol_prim[3, :],
            self.sol_domain.sol_int.sol_cons[0, :],
            self.sol_domain.sol_int.sol_cons[1, :],
            self.sol_domain.sol_int.sol_cons[2, :],
            self.sol_domain.sol_int.sol_cons[3, :],
            self.sol_domain.sol_int.reaction_source[0, :],
            self.sol_domain.sol_int.heat_release,
        ]
        vis_y_bounds = [
            [0.8e6, 1.1e6],
            [0.5, 2.5],
            [900, 1300],
            [0.3, 0.7],
            [1.8, 2.6],
            [1, 6],
            [-2e7, -1.5e7],
            [0.5, 1.75],
            [-1, 1],
            [-1, 1],
        ]
        vis_x_bounds = [[None, None]] * len(vis_vars)

        start_idx = 0
        for var_idx in range(len(vis_vars)):

            if var_idx >= 9:
                start_idx += 1

            plot = FieldPlot(
                self.solver.image_output_dir,
                0,
                1,
                self.solver.num_steps,
                self.solver.sim_type,
                vis_vars[start_idx : var_idx + 1],
                vis_x_bounds[start_idx : var_idx + 1],
                vis_y_bounds[start_idx : var_idx + 1],
                self.sol_domain.gas_model.species_names,
            )

            plot.plot(self.sol_domain, "b-", True)

            # check that bounds and values were set correctly
            if var_idx == 0:
                self.assertTrue(np.array_equal(vis_y_bounds[0], plot.ax.get_ylim()))
                self.assertTrue(np.array_equal(vis_vals[0], plot.ax.lines[0].get_xydata()[:, 1]))
            else:
                for ax_idx in range(var_idx - start_idx):
                    ax = plot.ax.ravel()[ax_idx]
                    self.assertTrue(np.array_equal(vis_y_bounds[ax_idx + start_idx], ax.get_ylim()))
                    self.assertTrue(np.array_equal(vis_vals[ax_idx + start_idx], ax.lines[0].get_xydata()[:, 1]))

            plot.fig.clf()

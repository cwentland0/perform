import unittest

import numpy as np

from constants import TEST_DIR, del_test_dir, gen_test_dir, solution_domain_setup
from perform.constants import REAL_TYPE
from perform.input_funcs import catch_list
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.probe_plot import ProbePlot


class ProbePlotInitTestCase(unittest.TestCase):
    def setUp(self):

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup(TEST_DIR)

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

        plot = ProbePlot(
            self.solver.image_output_dir,
            0,
            self.solver.sim_type,
            self.solver.probe_vars,
            self.vis_vars,
            1,
            self.solver.num_probes,
            self.vis_x_bounds,
            self.vis_y_bounds,
            self.sol_domain.gas_model.species_names,
        )

        self.assertEqual(plot.vis_type, "probe")
        self.assertEqual(plot.num_subplots, 4)
        self.assertEqual(plot.num_rows, 2)
        self.assertEqual(plot.num_cols, 2)

        for bound_idx, bound in enumerate(self.vis_x_bounds):
            self.assertTrue(np.array_equal(bound, plot.vis_x_bounds[bound_idx]))
            self.assertTrue(np.array_equal(self.vis_y_bounds[bound_idx], plot.vis_y_bounds[bound_idx]))


class ProbePlotMethodsTestCase(unittest.TestCase):
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

    def test_probe_plot_plot(self):

        sol_domain = self.sol_domain

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
        ]
        vis_x_bounds = [[None, None]] * len(vis_vars)

        # overwrite SolutionDomain probe info
        sol_domain.probe_vars = vis_vars
        sol_domain.num_probe_vars = len(vis_vars)
        self.solver.probe_vars = vis_vars

        # loop probes
        for probe_idx in range(sol_domain.num_probes):

            start_idx = 0
            # loop probe vars
            for var_idx in range(len(vis_vars)):

                # reset probe values
                sol_domain.probe_vals = np.zeros(
                    (sol_domain.num_probes, sol_domain.num_probe_vars, self.solver.num_steps), dtype=REAL_TYPE
                )

                if var_idx >= 9:
                    start_idx += 1

                plot = ProbePlot(
                    self.solver.image_output_dir,
                    0,
                    self.solver.sim_type,
                    sol_domain.probe_vars,
                    vis_vars[start_idx : var_idx + 1],
                    probe_idx,
                    self.solver.num_probes,
                    vis_x_bounds[start_idx : var_idx + 1],
                    vis_y_bounds[start_idx : var_idx + 1],
                    sol_domain.gas_model.species_names,
                )

                # update probes
                for self.solver.iter in range(self.solver.num_steps):
                    sol_domain.update_probes(self.solver)

                # plot
                plot.plot(sol_domain.probe_vals, sol_domain.time_vals, self.solver.num_steps, "b-", True)

                # check that bounds and values were set correctly
                if var_idx == 0:
                    self.assertTrue(np.array_equal(vis_y_bounds[0], plot.ax.get_ylim()))
                    self.assertTrue(
                        np.array_equal(sol_domain.probe_vals[probe_idx, 0, :], plot.ax.lines[0].get_xydata()[:, 1])
                    )
                else:
                    for ax_idx in range(var_idx - start_idx):
                        ax = plot.ax.ravel()[ax_idx]
                        self.assertTrue(np.array_equal(vis_y_bounds[ax_idx + start_idx], ax.get_ylim()))
                        self.assertTrue(
                            np.array_equal(
                                sol_domain.probe_vals[probe_idx, ax_idx + start_idx, :], ax.lines[0].get_xydata()[:, 1]
                            )
                        )

                plot.fig.clf()

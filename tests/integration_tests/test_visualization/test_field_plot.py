import unittest
import os
import shutil

import numpy as np

from constants import solution_domain_setup
from perform.input_funcs import catch_list
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.visualization.field_plot import FieldPlot


class FieldPlotInitTestCase(unittest.TestCase):
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

        # get some plotting variables
        param_dict = self.solver.param_dict
        self.vis_vars = catch_list(param_dict, "vis_var_0", [None])
        self.vis_x_bounds = catch_list(param_dict, "vis_x_bounds_0", [[None, None]], len_highest=len(self.vis_vars))
        self.vis_y_bounds = catch_list(param_dict, "vis_y_bounds_0", [[None, None]], len_highest=len(self.vis_vars))

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

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

        # get some plotting variables
        param_dict = self.solver.param_dict
        self.vis_vars = catch_list(param_dict, "vis_var_0", [None])
        self.vis_x_bounds = catch_list(param_dict, "vis_x_bounds_0", [[None, None]], len_highest=len(self.vis_vars))
        self.vis_y_bounds = catch_list(param_dict, "vis_y_bounds_0", [[None, None]], len_highest=len(self.vis_vars))

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_field_plot_plot(self):

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

        plot.plot(self.sol_domain, "b-", True)

        # for ax_idx, axis in plot.ax.ravel():

from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from perform.constants import FIG_WIDTH_DEFAULT, FIG_HEIGHT_DEFAULT
from perform.visualization.field_plot import FieldPlot
from perform.visualization.probe_plot import ProbePlot
from perform.input_funcs import catch_input, catch_list


class VisualizationGroup:
    """
    Container class for all visualizations
    """

    def __init__(self, sol_domain, solver):

        param_dict = solver.param_dict

        self.vis_show = catch_input(param_dict, "vis_show", True)
        self.vis_save = catch_input(param_dict, "vis_save", False)
        self.vis_interval = catch_input(param_dict, "vis_interval", 1)
        self.nice_vis = catch_input(param_dict, "nice_vis", False)

        # If not saving or showing, don't even draw the plots
        self.vis_draw = True
        if (not self.vis_show) and (not self.vis_save):
            self.vis_draw = False
            return

        # Count number of visualizations requested
        self.num_vis_plots = 0
        plot_count = True
        while plot_count:
            try:
                key_name = "vis_type_" + str(self.num_vis_plots + 1)
                plot_type = str(param_dict[key_name])
                # TODO: should honestly just fail for incorrect input
                assert (plot_type in ["field", "probe", "residual"]), (
                    key_name + " must be either"
                    + " \"field\", \"probe\", or \"residual\"")
                self.num_vis_plots += 1
            except (KeyError, AssertionError):
                plot_count = False

        if self.num_vis_plots == 0:
            print("WARNING: No visualization plots selected...")
            sleep(1.0)
        self.vis_list = [None] * self.num_vis_plots

        # Initialize each figure object
        for vis_idx in range(1, self.num_vis_plots + 1):

            # some parameters all plots have
            vis_type = str(param_dict["vis_type_" + str(vis_idx)])
            vis_vars = catch_list(
                param_dict, "vis_var_" + str(vis_idx), [None])
            vis_x_bounds = catch_list(
                param_dict, "vis_x_bounds_" + str(vis_idx),
                [[None, None]], len_highest=len(vis_vars))
            vis_y_bounds = catch_list(
                param_dict, "vis_y_bounds_" + str(vis_idx),
                [[None, None]], len_highest=len(vis_vars))

            if vis_type == "field":
                self.vis_list[vis_idx - 1] = FieldPlot(
                    solver.image_output_dir, vis_idx,
                    self.vis_interval, solver.num_steps,
                    solver.sim_type, vis_vars, vis_x_bounds,
                    vis_y_bounds, sol_domain.gas_model.species_names)

            elif vis_type == "probe":
                probe_num = -1 + catch_input(
                    param_dict,
                    "probe_num_" + str(vis_idx), -2)

                self.vis_list[vis_idx - 1] = ProbePlot(
                    solver.image_output_dir, vis_idx,
                    solver.sim_type, solver.probe_vars,
                    vis_vars, probe_num, solver.num_probes,
                    vis_x_bounds, vis_y_bounds,
                    sol_domain.gas_model.species_names)

            elif (vis_type == "residual"):
                raise ValueError("Residual plot not implemented yet")

            else:
                raise ValueError("Invalid visualization type: " + vis_type)

        # Set plot positions/dimensions
        for vis in self.vis_list:
            vis.fig, vis.ax = plt.subplots(
                nrows=vis.num_rows, ncols=vis.num_cols,
                num=vis.vis_id,
                figsize=(FIG_WIDTH_DEFAULT, FIG_HEIGHT_DEFAULT))

        if self.vis_show:
            plt.show(block=False)
            plt.pause(0.001)

    def draw_plots(self, sol_domain, solver):
        """
        Helper function to draw, display, and save plots
        """

        if not self.vis_draw:
            return

        if self.num_vis_plots > 0:
            if (solver.iter % self.vis_interval) != 0:
                return

            # Decide whether to draw plot for first time or update y-data
            if int(solver.iter / self.vis_interval) == 1:
                first_plot = True
            else:
                first_plot = False

            for vis in self.vis_list:

                # draw and save plots
                if vis.vis_type == "field":
                    vis.plot(
                        sol_domain.sol_int.sol_prim,
                        sol_domain.sol_int.sol_cons,
                        sol_domain.sol_int.source, sol_domain.sol_int.rhs,
                        sol_domain.gas_model, sol_domain.mesh.x_cell, 'b-',
                        first_plot)

                elif vis.vis_type == "probe":
                    vis.plot(
                        sol_domain.probe_vals, sol_domain.time_vals,
                        solver.iter, 'b-', first_plot)
                else:
                    raise ValueError("Invalid vis_type:" + str(vis.vis_type))
                if self.vis_save:
                    vis.save(solver.iter)

                if self.vis_show:
                    vis.fig.canvas.flush_events()

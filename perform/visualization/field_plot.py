import os
from math import floor, log

import numpy as np
import matplotlib.pyplot as plt

from perform.visualization.visualization import Visualization


class FieldPlot(Visualization):
    """Class for field plot visualization.

    Produces "snapshots" of flow field profiles at the interval specified by vis_interval.

    Must be manually expanded to plot specific profiles.

    Args:
        image_output_dir: Base image output directory, within the working directory.
        vis_id: Zero-indexed ID of this Visualization.
        vis_interval: Physical time step interval at which to display/save visualization plots.
        num_steps: Total number of physical time steps to be executed by this simulation.
        sim_type: "FOM" or "ROM", depending on simulation type.
        vis_vars: List of num_subplots strings of variables to be visualized.
        vis_x_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            x-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        vis_y_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            y-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        species_names: List of strings of names for all num_species_full chemical species.

    Attributes:
        vis_type: Set to "field".
        x_label: Set to "x (m)", x-axis label.
        num_imgs:
            Total number of field plot images that are expected to be generated, under the assumption
            that the simulation completes without failure.
        img_string: String within which the output image number will be inserted for image file name generation.
        ax_line: List of matplotlib.lines.Line2D artists for each subplot.
        img_dir: Output director where image files will be saved.
    """

    def __init__(
        self,
        image_output_dir,
        vis_id,
        vis_interval,
        num_steps,
        sim_type,
        vis_vars,
        vis_x_bounds,
        vis_y_bounds,
        species_names,
    ):

        self.vis_type = "field"
        self.vis_interval = vis_interval
        self.vis_id = vis_id
        self.x_label = "x (m)"

        assert vis_vars[0] is not None, "Must provide vis_vars for field plot " + str(self.vis_id)

        self.num_imgs = int(num_steps / vis_interval)
        if self.num_imgs > 0:
            self.img_string = "%0" + str(floor(log(self.num_imgs, 10)) + 1) + "d"
        else:
            self.img_string = None

        super().__init__(vis_id, vis_vars, vis_x_bounds, vis_y_bounds, species_names)

        self.ax_line = [None] * self.num_subplots

        # Set up output directory
        vis_name = ""
        for visVar in self.vis_vars:
            vis_name += "_" + visVar
        vis_name += "_" + sim_type
        self.img_dir = os.path.join(image_output_dir, "field" + vis_name)
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)

    def plot(self, sol_prim, sol_cons, source, rhs, gas, x_cell, line_style, first_plot):
        """Draw and display field plot.

        Saves a decent amount of time by using set_ydata instead of repeatedly clearing axes and replotting.

        Args:
            sol_prim: NumPy array of the primitive state profiles from SolutionInterior.
            sol_cons: NumPy array of the conservative state profiles from SolutionInterior.
            source: NumPy array of the reaction source term profiles for the num_species species transport equations.
            rhs: NumPy array of the evaluation of the right-hand side function of the semi-discrete governing ODE.
            gas: GasModel object associated with the SolutionDomain.
            x_cell: Coordinates of Mesh cell centers.
            line_style: String containing matplotlib.pyplot line style option.
            first_plot: Boolean flag indicating whether this is the first time the plot is being drawn.
        """

        plt.figure(self.vis_id)

        # TODO: Just aggregate axes into a list during __init__
        if not isinstance(self.ax, np.ndarray):
            ax_list = [self.ax]
        else:
            ax_list = self.ax

        for col_idx, col in enumerate(ax_list):
            if not isinstance(col, np.ndarray):
                col_list = [col]
            else:
                col_list = col
            for row_idx, ax_var in enumerate(col_list):

                lin_idx = np.ravel_multi_index(([col_idx], [row_idx]), (self.num_rows, self.num_cols))[0]
                if (lin_idx + 1) > self.num_subplots:
                    ax_var.axis("off")
                    continue

                y_data = self.get_y_data(sol_prim, sol_cons, source, rhs, self.vis_vars[lin_idx], gas)

                if first_plot:
                    x_data = x_cell
                    (self.ax_line[lin_idx],) = ax_var.plot(x_data, y_data, line_style)
                    ax_var.set_ylabel(self.ax_labels[lin_idx])
                    ax_var.set_xlabel(self.x_label)
                else:
                    self.ax_line[lin_idx].set_ydata(y_data)

                ax_var.relim()
                ax_var.autoscale_view()
                ax_var.set_ylim(bottom=self.vis_y_bounds[lin_idx][0], top=self.vis_y_bounds[lin_idx][1], auto=True)
                ax_var.set_xlim(left=self.vis_x_bounds[lin_idx][0], right=self.vis_x_bounds[lin_idx][1], auto=True)

                ax_var.ticklabel_format(useOffset=False)

        if first_plot:
            self.fig.tight_layout()
        self.fig.canvas.draw()

    def get_y_data(self, sol_prim, sol_cons, source, rhs, var_str, gas):
        """Extract plotting data from flow field domain data.

        New inputs and var_str options must be manually added to permit new profiles to be plotted.

        Args:
            sol_prim: NumPy array of the primitive state profiles from SolutionInterior.
            sol_cons: NumPy array of the conservative state profiles from SolutionInterior.
            source: NumPy array of the reaction source term profiles for the num_species species transport equations.
            rhs: NumPy array of the evaluation of the right-hand side function of the semi-discrete governing ODE.
            var_str: String corresponding to variable profile to be plotted.
            gas: GasModel object associated with the SolutionDomain.

        Returns:
            NumPy array of the flow field profile to be visualized.
        """

        if var_str == "pressure":
            y_data = sol_prim[0, :]
        elif var_str == "velocity":
            y_data = sol_prim[1, :]
        elif var_str == "temperature":
            y_data = sol_prim[2, :]
        elif var_str == "source":
            y_data = source[0, :]
        elif var_str == "density":
            y_data = sol_cons[0, :]
        elif var_str == "momentum":
            y_data = sol_cons[1, :]
        elif var_str == "energy":
            y_data = sol_cons[2, :]
        elif var_str[:7] == "species":
            spec_idx = int(var_str[8:])
            if spec_idx == gas.num_species_full:
                massFracs = gas.calc_all_mass_fracs(sol_prim[3:, :], threshold=False)
                y_data = massFracs[-1, :]
            else:
                y_data = sol_prim[3 + spec_idx - 1, :]
        elif var_str[:15] == "density-species":
            spec_idx = int(var_str[16:])
            y_data = sol_cons[3 + spec_idx - 1, :]
        else:
            raise ValueError("Invalid field visualization variable:" + str(var_str))

        return y_data

    def save(self, iter_num, dpi=100):
        """Save plot to disk.

        Args:
            iter_num: One-indexed simulation iteration number.
            dpi: Dots per inch of figure, determines resolution.
        """

        plt.figure(self.vis_id)
        visIdx = int(iter_num / self.vis_interval)
        fig_num = self.img_string % visIdx
        fig_file = os.path.join(self.img_dir, "fig_" + fig_num + ".png")
        self.fig.savefig(fig_file, dpi=dpi)

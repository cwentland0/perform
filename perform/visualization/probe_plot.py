import os

import matplotlib as mpl
if os.environ["PLT_USE_AGG"] == "1":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from perform.visualization.visualization import Visualization

# TODO: maybe easier to make probe/residual plots under some PointPlot class
# TODO: move some of the init input arguments used for assertions outside


class ProbePlot(Visualization):
    """Class for probe plot visualization.

    Produces plots of the time history of probe monitors at the interval specified by vis_interval.

    Must be manually expanded to plot specific profiles.

    Args:
        image_output_dir: Base image output directory, within the working directory.
        vis_id: Zero-indexed ID of this Visualization.
        sim_type: "FOM" or "ROM", depending on simulation type.
        probe_vars: List of strings of variables to be probed at each probe monitor location.
        vis_vars: List of num_subplots strings of variables to be visualized.
        probe_num: One-indexed probe number.
        num_probes: Total number of probe monitors set in the solver parameters input file
        vis_x_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            x-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        vis_y_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            y-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        species_names: List of strings of names for all num_species_full chemical species.

    Attributes:
        vis_type: Set to "probe".
        x_label: Set to "t (s)", x-axis label.
        probe_num: One-indexed probe number, corresponding to labels given in the solver parameters input file.
        probe_vars: List of strings of variables to be probed at each probe monitor location.
        ax_line: List of matplotlib.lines.Line2D artists for each subplot.
        fig_file: Path to visualization plot output image file.
    """

    def __init__(
        self,
        image_output_dir,
        vis_id,
        sim_type,
        probe_vars,
        vis_vars,
        probe_num,
        num_probes,
        vis_x_bounds,
        vis_y_bounds,
        species_names,
    ):

        self.vis_type = "probe"
        self.vis_id = vis_id
        self.x_label = "t (s)"

        self.probe_num = probe_num
        self.probe_vars = probe_vars
        assert self.probe_num > 0, "Must provide positive integer probe number for probe plot " + str(self.vis_id)
        assert self.probe_num <= num_probes, "probe_num_" + str(self.vis_id) + " must correspond to a valid probe"
        assert vis_vars[0] is not None, "Must provide vis_vars for probe plot " + str(self.vis_id)

        super().__init__(vis_id, vis_vars, vis_x_bounds, vis_y_bounds, species_names)

        self.ax_line = [None] * self.num_subplots

        # Image file on disk
        vis_name = ""
        for vis_var in self.vis_vars:
            vis_name += "_" + vis_var
        fig_name = "probe" + vis_name + "_" + str(self.probe_num) + "_" + sim_type + ".png"
        self.fig_file = os.path.join(image_output_dir, fig_name)

        # Check that requested variables are being probed
        for vis_var in self.vis_vars:
            assert vis_var in probe_vars, "Must probe " + vis_var + " to plot it"

    def plot(self, probe_vals, time_vals, iter_num, line_style, first_plot):
        """Draw and display probe plot.

        Since shape of plotted data changes every time, can't use set_data.
        As a result, probe plotting can be pretty slow.

        Args:
            probe_vals: NumPy array of probe monitor time history for each monitored variable.
            time_vals: NumPy array of time values associated with each discrete time step, not including t = 0.
            iter_num: One-indexed physical time step iteration number.
            line_style: String containing matplotlib.pyplot line style option.
            first_plot: Boolean flag indicating whether this is the first time the plot is being drawn.
        """

        plt.figure(self.vis_id)

        if type(self.ax) != np.ndarray:
            ax_list = [self.ax]
        else:
            ax_list = self.ax

        for col_idx, col in enumerate(ax_list):
            if type(col) != np.ndarray:
                col_list = [col]
            else:
                col_list = col
            for rowIdx, ax_var in enumerate(col_list):

                lin_idx = np.ravel_multi_index(([col_idx], [rowIdx]), (self.num_rows, self.num_cols))[0]
                if (lin_idx + 1) > self.num_subplots:
                    ax_var.axis("off")
                    continue

                ax_var.cla()

                y_data = self.get_y_data(probe_vals, self.vis_vars[lin_idx], iter_num)
                x_data = time_vals[:iter_num]

                (self.ax_line[lin_idx],) = ax_var.plot(x_data, y_data, line_style)

                ax_var.set_ylabel(self.ax_labels[lin_idx])
                ax_var.set_xlabel(self.x_label)
                ax_var.set_ylim(bottom=self.vis_y_bounds[lin_idx][0], top=self.vis_y_bounds[lin_idx][1], auto=True)
                ax_var.set_xlim(left=self.vis_x_bounds[lin_idx][0], right=self.vis_x_bounds[lin_idx][1], auto=True)
                ax_var.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if first_plot:
            self.fig.tight_layout()
        self.fig.canvas.draw()

    def get_y_data(self, probe_vals, var_str, iter_num):
        """Extract probe data to be plotted from probe_vals.

        Data extraction of probe monitor data is done in SolutionDomain, this just selects the correct variable.

        Args:
            probe_vals: NumPy array of probe monitor time history for each monitored variable.
            var_str: String corresponding to variable profile to be plotted.
            iter_num: One-indexed simulation iteration number.
        """

        var_idx = np.squeeze(np.argwhere(self.probe_vars == var_str)[0])
        y_data = probe_vals[self.probe_num - 1, var_idx, :iter_num]

        return y_data

    def save(self, iter_num, dpi=100):
        """Save plot to disk.

        Args:
            iter_num: One-indexed simulation iteration number.
            dpi: Dots per inch of figure, determines resolution.
        """

        plt.figure(self.vis_id)
        self.fig.savefig(self.fig_file, dpi=dpi)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec

mpl.rc("font", family="serif", size="10")
mpl.rc("axes", labelsize="x-large")
mpl.rc("figure", facecolor="w")
mpl.rc("text", usetex=False)
mpl.rc("text.latex", preamble=r"\usepackage{amsmath}")

# TODO: adapt axis label font size to number of subplots
# TODO: add RHS, flux plotting


class Visualization:
    """Base class for visualization plots.

    Processes input parameters for plotting, such as axis bounds, arrangement of subplots, and axis labels.

    Child classes must implement the plot() and save() member functions.

    Args:
        vis_id: Zero-indexed ID of this Visualization.
        vis_vars: List of num_subplots strings of variables to be visualized.
        vis_x_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            x-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        vis_y_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            y-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        species_names: List of strings of names for all num_species_full chemical species.
        legend_labels: List of strings for legend labels.

    Attributes:
        species_names: List of strings of names for all num_species_full chemical species.
        legend_labels: List of strings for legend labels.
        vis_type: String indicating the type of plot that this Visualization will represent, e.g. "field"
        num_subplots: Number of subplots within figure.
        vis_x_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            x-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        vis_y_bounds:
            List of num_subplots lists, each of length 2 including lower and upper bounds on the
            y-axis for each subplot. Setting to [None, None] allows for dynamic axis sizing.
        num_rows: Number of rows in subplot grid within figure.
        num_cols: Number of columns in subplot grid within figure.
        ax_labels: List of num_subplots strings for label of each subplot's y-axis.
    """

    def __init__(self, vis_id, vis_vars, vis_x_bounds, vis_y_bounds, species_names, legend_labels=None):

        self.species_names = species_names
        num_species_full = len(species_names)
        self.legend_labels = legend_labels

        if self.vis_type in ["field", "probe"]:

            # Check requested variables
            self.vis_vars = vis_vars
            for vis_var in self.vis_vars:
                if vis_var in ["pressure", "velocity", "temperature", "source", "density", "momentum", "energy"]:
                    pass
                elif (vis_var[:7] == "species") or (vis_var[:15] == "density-species"):
                    try:
                        if vis_var[:7] == "species":
                            species_idx = int(vis_var[8:])
                        elif vis_var[:15] == "density-species":
                            species_idx = int(vis_var[16:])

                        assert (species_idx > 0) and (species_idx <= num_species_full), (
                            "Species number must be a positive integer " + "<= the number of chemical species"
                        )
                    except ValueError:
                        raise ValueError(
                            "vis_var entry "
                            + vis_var
                            + " must be formated as species_X "
                            + "or density-species_X, where X is an integer"
                        )
                else:
                    raise ValueError("Invalid entry in vis_var" + str(vis_id))

            self.num_subplots = len(self.vis_vars)

        else:
            raise ValueError("Invalid value for vis_type: " + str(self.vis_type))

        self.vis_x_bounds = vis_x_bounds
        assert len(self.vis_x_bounds) == self.num_subplots, (
            "Length of vis_x_bounds" + str(vis_id) + " must match number of subplots: " + str(self.num_subplots)
        )
        self.vis_y_bounds = vis_y_bounds
        assert len(self.vis_y_bounds) == self.num_subplots, (
            "Length of vis_y_bounds" + str(vis_id) + " must match number of subplots: " + str(self.num_subplots)
        )

        if self.num_subplots == 1:
            self.num_rows = 1
            self.num_cols = 1
        elif self.num_subplots == 2:
            self.num_rows = 2
            self.num_cols = 1
        elif self.num_subplots <= 4:
            self.num_rows = 2
            self.num_cols = 2
        elif self.num_subplots <= 6:
            self.num_rows = 3
            self.num_cols = 2
        elif self.num_subplots <= 9:
            self.num_rows = 3
            self.num_cols = 3
        else:
            raise ValueError("Cannot plot more than nine" + " subplots in the same image")

        # Axis labels
        # TODO: could change this to a dictionary reference
        self.ax_labels = [None] * self.num_subplots
        if self.vis_type == "residual":
            self.ax_labels[0] = "Residual History"
        else:
            for ax_idx in range(self.num_subplots):
                var_str = self.vis_vars[ax_idx]
                if var_str == "pressure":
                    self.ax_labels[ax_idx] = r"Pressure $\left( Pa \right)$"
                elif var_str == "velocity":
                    self.ax_labels[ax_idx] = r"Velocity $\left( \frac{m}{s} \right)$"
                elif var_str == "temperature":
                    self.ax_labels[ax_idx] = r"Temperature $\left( K \right)$"
                elif var_str == "source":
                    self.ax_labels[ax_idx] = r"Source Term $\left( \frac{kg}{m^3 \; s} \right)$"
                elif var_str == "density":
                    self.ax_labels[ax_idx] = r"Density $\left( \frac{kg}{m^3} \right)$"
                elif var_str == "momentum":
                    self.ax_labels[ax_idx] = r"Momentum $\left( \frac{kg}{s \; m^2} \right)$"
                elif var_str == "energy":
                    self.ax_labels[ax_idx] = r"Energy $\left( \frac{J}{m^3} \right)$"
                elif var_str[:7] == "species":
                    self.ax_labels[ax_idx] = r"$Y_{%s}$" % (self.species_names[int(var_str[8:]) - 1])
                elif var_str[:15] == "density-species":
                    self.ax_labels[ax_idx] = r"$\rho Y_{%s}$" % (self.species_names[int(var_str[16:]) - 1])
                else:
                    raise ValueError("Invalid field visualization variable:" + str(var_str))

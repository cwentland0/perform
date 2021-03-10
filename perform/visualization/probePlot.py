import os

import matplotlib.pyplot as plt
import numpy as np

import perform.constants as const
from perform.visualization.visualization import Visualization

# TODO: might be easier to make probe and residual plots under some pointPlot class
# TODO: move some of the init input arguments used for assertions outside

class ProbePlot(Visualization):

	def __init__(self, vis_id, sim_type, probe_vars, vis_vars, probe_num, num_probes, vis_x_bounds, vis_y_bounds, species_names):

		self.vis_type = "probe"
		self.vis_id   = vis_id
		self.x_label  = "t (s)"

		self.probe_num  = probe_num
		self.probe_vars = probe_vars
		assert (self.probe_num >= 0 ), "Must provide positive integer probe number for probe plot "+str(self.vis_id)
		assert (self.probe_num < num_probes), "probe_num_"+str(self.vis_id)+" must correspond to a valid probe"
		assert (vis_vars[0] is not None), "Must provide vis_vars for probe plot "+str(self.vis_id)

		super().__init__(vis_id, vis_vars, vis_x_bounds, vis_y_bounds, species_names)

		# image file on disk
		vis_name = ""
		for vis_var in self.vis_vars:
			vis_name += "_"+vis_var
		fig_name = "probe" + vis_name + "_" + str(self.probe_num) + "_" + sim_type + ".png"
		self.fig_file = os.path.join(const.image_output_dir, fig_name) 

		# check that requested variables are being probed
		for vis_var in self.vis_vars:
			assert (vis_var in probe_vars), "Must probe "+vis_var+" to plot it"


	def plot(self, probe_vals, time_vals, iter_num, line_style):
		"""
		Draw and display plot
		"""

		plt.figure(self.vis_id)

		if (type(self.ax) != np.ndarray):
			ax_list = [self.ax]
		else:
			ax_list = self.ax

		for col_idx, col in enumerate(ax_list):
			if (type(col) != np.ndarray):
				col_list = [col]
			else:
				col_list = col
			for rowIdx, ax_var in enumerate(col_list):

				lin_idx = np.ravel_multi_index(([col_idx],[rowIdx]), (self.num_rows, self.num_cols))[0]
				if ((lin_idx+1) > self.num_subplots): 
					ax_var.axis("off")
					break

				y_data = self.getYData(probe_vals, self.vis_vars[lin_idx], iter_num)
				x_data = time_vals[:iter_num]

				ax_var.plot(x_data, y_data, line_style)
				ax_var.set_ylim(self.vis_y_bounds[lin_idx])
				ax_var.set_xlim(self.vis_x_bounds[lin_idx])
				ax_var.set_ylabel(self.ax_labels[lin_idx])
				ax_var.set_xlabel(self.x_label)
				
				ax_var.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

		self.fig.tight_layout()
		self.fig.canvas.draw_idle()


	def getYData(self, probe_vals, var_str, iter_num):

		# data extraction of probes is done in solDomain 
		var_idx = np.squeeze(np.argwhere(self.probe_vars == var_str)[0])
		y_data = probe_vals[self.probe_num, var_idx, :iter_num]

		return y_data


	def save(self, iter_num, dpi=100):

		plt.figure(self.vis_id)
		plt.tight_layout()
		self.fig.savefig(self.fig_file, dpi=dpi)

		
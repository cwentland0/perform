import os
from math import floor, log

import numpy as np
import matplotlib.pyplot as plt

import perform.constants as const
from perform.visualization.visualization import Visualization


class FieldPlot(Visualization):
	"""
	Class for field plot image
	"""

	def __init__(self, vis_id, vis_interval, num_steps, sim_type, vis_vars, vis_x_bounds, vis_y_bounds, species_names):

		self.vis_type        = "field"
		self.vis_interval   = vis_interval
		self.vis_id         = vis_id
		self.x_label         = "x (m)"

		self.num_imgs = int(num_steps / vis_interval)
		if (self.num_imgs > 0):
			self.img_string 	= '%0'+str(floor(log(self.num_imgs, 10))+1)+'d'
		else:
			self.img_string 	= None

		super().__init__(vis_id, vis_vars, vis_x_bounds, vis_y_bounds, species_names)

		# Set up output directory
		vis_name = ""
		for visVar in self.vis_vars:
			vis_name += "_" + visVar
		vis_name += "_" + sim_type
		self.img_dir = os.path.join(const.image_output_dir, "field"+vis_name)
		if not os.path.isdir(self.img_dir): os.mkdir(self.img_dir)
		

	def plot(self, sol_prim, sol_cons, source, rhs, gas, x_cell, line_style):
		"""
		Draw and display plot
		"""

		plt.figure(self.vis_id)

		# TODO: Just aggregate axes into a list during __init__
		if (type(self.ax) != np.ndarray):
			ax_list = [self.ax]
		else:
			ax_list = self.ax

		for col_idx, col in enumerate(ax_list):
			if (type(col) != np.ndarray):
				col_list = [col]
			else:
				col_list = col
			for row_idx, ax_var in enumerate(col_list):

				lin_idx = np.ravel_multi_index(([col_idx],[row_idx]), (self.num_rows, self.num_cols))[0]
				if ((lin_idx+1) > self.numSubplots): 
					ax_var.axis("off")
					break

				
				y_data = self.get_y_data(sol_prim, sol_cons, source, rhs, self.vis_vars[lin_idx], gas)
				x_data = x_cell

				ax_var.plot(x_data, y_data, line_style)
				ax_var.set_ylim(self.vis_y_bounds[lin_idx])
				ax_var.set_xlim(self.vis_x_bounds[lin_idx])
				ax_var.set_ylabel(self.ax_labels[lin_idx])
				ax_var.set_xlabel(self.x_label)
				
				if (self.vis_type == "field"):
					ax_var.ticklabel_format(useOffset=False)
				else:
					ax_var.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

		self.fig.tight_layout()
		self.fig.canvas.draw_idle()


	def get_y_data(self, sol_prim, sol_cons, source, rhs, var_str, gas):
		"""
		Extract plotting data from flow field domain data
		"""

		try:
			if (var_str == "pressure"):
				y_data = sol_prim[0,:]
			elif (var_str == "velocity"):
				y_data = sol_prim[1,:]
			elif (var_str == "temperature"):
				y_data = sol_prim[2,:]
			elif (var_str == "source"):
				y_data = source[0,:]
			elif (var_str == "density"):
				y_data = sol_cons[0,:]
			elif (var_str == "momentum"):
				y_data = sol_cons[1,:]
			elif (var_str == "energy"):
				y_data = sol_cons[2,:]
			elif (var_str[:7] == "species"):
				spec_idx = int(var_str[7:])
				if (spec_idx == gas.num_species_full):
					massFracs = gas.calc_all_mass_fracs(sol_prim[3:,:], threshold=False)
					y_data = massFracs[-1,:]
				else:
					y_data = sol_prim[3+spec_idx-1,:]

			# TODO: Get density-species for last species
			elif (var_str[:15] == "density-species"):
				spec_idx = int(var_str[15:])
				y_data = sol_cons[3+spec_idx-1,:]
		except Exception as e:
			print(e)
			raise ValueError("Invalid field visualization variable:"+str(var_str))

		return y_data


	def save(self, iter_num, dpi=100):
		"""
		Save plot to disk
		"""

		plt.figure(self.vis_id)
		plt.tight_layout()
		visIdx 	= int(iter_num / self.vis_interval)
		fig_num 	= self.img_string % visIdx
		fig_file = os.path.join(self.img_dir, "fig_"+fig_num+".png")
		self.fig.savefig(fig_file, dpi=dpi)
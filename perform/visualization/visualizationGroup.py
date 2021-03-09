from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from perform.constants import fig_width_default, fig_height_default
from perform.visualization.field_plot import FieldPlot
from perform.visualization.probe_plot import ProbePlot
from perform.input_funcs import catch_input, catch_list


class VisualizationGroup:
	"""
	Container class for all visualizations
	"""

	def __init__(self, sol_domain, solver):
		
		param_dict = solver.param_dict

		self.vis_show 		= catch_input(param_dict, "vis_show", True)
		self.vis_save 		= catch_input(param_dict, "vis_save", False)
		self.vis_interval 	= catch_input(param_dict, "vis_interval", 1)
		self.nice_vis 		= catch_input(param_dict, "nice_vis", False)

		# if not saving or showing, don't even draw the plots
		self.vis_draw = True
		if ((not self.vis_show) and (not self.vis_save)):
			self.vis_draw = False
			return

		# count number of visualizations requested
		self.num_vis_plots = 0
		plot_count = True
		while plot_count:
			try:
				key_name = "vis_type"+str(self.num_vis_plots+1)
				plot_type = str(param_dict[key_name])
				# TODO: should honestly just fail for incorrect input
				assert (plot_type in ["field", "probe", "residual"]), (key_name+" must be either \"field\", \"probe\", or \"residual\"")
				self.num_vis_plots += 1
			except:
				plot_count = False

		if (self.num_vis_plots == 0):
			print("WARNING: No visualization plots selected...")
			sleep(1.0)
		self.vis_list = [None] * self.num_vis_plots

		# initialize each figure object
		for vis_idx in range(1, self.num_vis_plots+1):
			
			# some parameters all plots have
			vis_type    = str(param_dict["vis_type"+str(vis_idx)])
			vis_vars	   = catch_list(param_dict, "visVar"+str(vis_idx), [None])
			vis_x_bounds = catch_list(param_dict, "vis_x_bounds"+str(vis_idx), [[None,None]], len_highest=len(vis_vars))
			vis_y_bounds = catch_list(param_dict, "vis_y_bounds"+str(vis_idx), [[None,None]], len_highest=len(vis_vars))

			if (vis_type == "field"):
				self.vis_list[vis_idx-1] = FieldPlot(vis_idx, self.vis_interval, solver.num_steps, solver.sim_type, 
											vis_vars, vis_x_bounds, vis_y_bounds, sol_domain.gas_model.species_names)
			elif (vis_type == "probe"):
				probe_num = catch_input(param_dict, "probe_num"+str(vis_idx), -2) - 1
				self.vis_list[vis_idx-1] = ProbePlot(vis_idx, solver.sim_type, solver.probeVars, vis_vars, probe_num, 
											solver.numProbes, vis_x_bounds, vis_y_bounds, sol_domain.gas_model.species_names)
			elif (vis_type == "residual"):
				raise ValueError("Residual plot not implemented yet")
			else:
				raise ValueError("Invalid visualization selection: "+vis_type)

		# set plot positions/dimensions
		if (self.nice_vis and self.vis_show):
			try:
				raise ValueError("nice_vis is broken right now, please set nice_vis = False")
				# self.movePlots()
			except:
				for vis in self.vis_list:
					vis.fig, vis.ax = plt.subplots(nrows=vis.num_rows, ncols=vis.num_cols, num=vis.vis_id, figsize=(fig_width_default,fig_height_default))
		else:
			for vis in self.vis_list:
				vis.fig, vis.ax = plt.subplots(nrows=vis.num_rows, ncols=vis.num_cols, num=vis.vis_id, figsize=(fig_width_default,fig_height_default))	
			
		if self.vis_show:
			plt.show(block=False)
			plt.pause(0.001)

	def drawPlots(self, sol_domain, solver):
		""" 
		Helper function to draw, display, and save plots
		"""

		if not self.vis_draw: return

		if (self.num_vis_plots > 0):
			if ((solver.iter % self.vis_interval) != 0):
				return

			for vis in self.vis_list:

				# clear plots
				plt.figure(vis.vis_id)
				if (type(vis.ax) != np.ndarray):
					ax_list = [vis.ax]
				else:
					ax_list = vis.ax
				for col_idx, col in enumerate(ax_list):
					if (type(col) != np.ndarray):
						col_list = [col]
					else:
						col_list = col
					for rowIdx, ax_var in enumerate(col_list):
						ax_var.cla()

				# draw and save plots
				if (vis.vis_type == "field"):
					vis.plot(sol_domain.sol_int.sol_prim, sol_domain.sol_int.sol_cons, sol_domain.sol_int.source, sol_domain.sol_int.rhs, 
							sol_domain.gas_model, solver.mesh.x_cell, 'b-')
				elif (vis.vis_type == "probe"):
					vis.plot(sol_domain.probe_vals, sol_domain.time_vals, solver.iter, 'b-')
				else:
					raise ValueError("Invalid vis_type:" + str(vis.vis_type))
				if self.vis_save: vis.save(solver.iter)

			if self.vis_show:
				plt.show(block=False)
				plt.pause(0.001)


	# def movePlots(self):
	# 	"""
	# 	Resizes and moves plots to positions in the window for better viewing
	# 	"""

	# 	backend = matplotlib.get_backend()

	# 	# check some options, do some math for plot placement
	# 	# TODO: this doesn't account for taskbar, overestimates the size of available screen size
	# 	fig = plt.figure(num=0)
	# 	dpi = fig.dpi
	# 	manager = plt.get_current_fig_manager()
	# 	if (backend == "TkAgg"):
	# 		manager.resize(*manager.window.maxsize())
	# 		plt.pause(0.01)
	# 		screenW = manager.window.winfo_width()
	# 		screenH = manager.window.winfo_height()
	# 	elif (backend == "WXAgg"):
	# 		import wx
	# 		screenW, screenH = wx.Display(0).GetGeometry().GetSize()
	# 	elif (backend in ["Qt4Agg","Qt5Agg"]):
	# 		# TODO: this is super sketchy and may give weird results
	# 		manager.full_screen_toggle()
	# 		plt.pause(0.01)
	# 		screenW = manager.canvas.width()
	# 		screenH = manager.canvas.height()
	# 	else:
	# 		raise ValueError("Nice plot positioning not supported for matplotlib backend "+backend)
	# 	plt.close(0)

	# 	self.numSubplotsArr = np.zeros(self.num_vis_plots, dtype=np.int32)
	# 	for vis_idx, vis in enumerate(self.vis_list):
	# 		self.numSubplotsArr[vis_idx] = vis.numSubplots

	# 	xBase = 50
	# 	yBase = 50
	# 	screenW -= xBase
	# 	screenH -= yBase

	# 	figX = [None] * self.num_vis_plots
	# 	figY = [None] * self.num_vis_plots
	# 	figW = [None] * self.num_vis_plots
	# 	figH = [None] * self.num_vis_plots
	# 	if (self.num_vis_plots == 1):
	# 		# just fill the screen
	# 		figX[0], figY[0] = 0.0, 0.0
	# 		figW[0] = screenW
	# 		figH[0] = screenH
	# 	elif (self.num_vis_plots == 2):
	# 		for figIdx in range(2):
	# 			figX[figIdx] = (figIdx / 2.0) * screenW
	# 			figY[figIdx] = 0.0
	# 			figW[figIdx] = screenW / 2.0
	# 			figH[figIdx] = screenH
	# 	elif (self.num_vis_plots == 3):
	# 		allEqual = np.all(self.numSubplotsArr == self.numSubplotsArr[0])
	# 		if allEqual:
	# 			for figIdx in range(3):
	# 				figW[figIdx] = screenW / 2.0
	# 				figH[figIdx] = screenH / 2.0
	# 				if (figIdx < 2):
	# 					figX[figIdx] = (figIdx / 2.0) * screenW
	# 					figY[figIdx] = 0.0
	# 				else:
	# 					figX[figIdx] = 0.0
	# 					figY[figIdx] = screenH / 2.0

	# 		else:
	# 			largestPlot = np.argmax(self.numSubplotsArr)
	# 			smallCounter = 0
	# 			for figIdx in range(3):
	# 				figW[figIdx] = screenW / 2.0
	# 				if (figIdx == largestPlot):
	# 					figH[figIdx] = screenH
	# 					figX[figIdx] = 0.0
	# 					figY[figIdx] = 0.0
	# 				else:
	# 					figH[figIdx] = screenH / 2.0
	# 					figY[figIdx] = (smallCounter / 2.0) * screenH
	# 					smallCounter += 1

	# 	elif (self.num_vis_plots == 4):
	# 		for figIdx in range(4):
	# 			xIdx = figIdx % 2
	# 			yIdx = int(figIdx / 2)
	# 			figH[figIdx] = screenH / 2.0
	# 			figW[figIdx] = screenW / 2.0

	# 	else:
	# 		raise ValueError("Nice plot position not supported for more than four plots")

	# 	# convert pixels to inches
	# 	figW = [x/dpi for x in figW]
	# 	figH = [x/dpi for x in figH]

	# 	# move plots
	# 	for vis_idx, vis in enumerate(self.vis_list):

	# 		vis.fig, vis.ax = plt.subplots(nrows=vis.num_rows, ncols=vis.num_cols, num=vis.vis_id, figsize=(figW[vis_idx],figH[vis_idx]))
	# 		window = vis.fig.canvas.manager.window
	# 		if (backend == "TkAgg"):
	# 			window.wm_geometry("+%d+%d" % (figX[vis_idx], figY[vis_idx]))
	# 		elif (backend == "WXAgg"):
	# 			window.SetPosition((figX[vis_idx], figY[vis_idx]))
	# 		elif (backend in ["Qt4Agg","Qt5Agg"]):
	# 			window.move(figX[vis_idx], figY[vis_idx])
			
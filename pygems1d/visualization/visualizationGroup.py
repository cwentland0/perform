from pygems1d.constants import figWidthDefault, figHeightDefault
from pygems1d.visualization.fieldPlot import fieldPlot
from pygems1d.visualization.probePlot import probePlot
# from pygems1d.visualization.residualPlot import residualPlot
from pygems1d.inputFuncs import catchInput

import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

class visualizationGroup:
	"""
	Container class for all visualizations
	"""

	def __init__(self, solver):
		
		paramDict = solver.paramDict

		self.visShow 		= catchInput(paramDict, "visShow", True)
		self.visSave 		= catchInput(paramDict, "visSave", False)
		self.visInterval 	= catchInput(paramDict, "visInterval", 1)
		self.niceVis 		= catchInput(paramDict, "niceVis", False)

		# count number of visualizations requested
		self.numVisPlots = 0
		plotCount = True
		while plotCount:
			try:
				keyName = "visType"+str(self.numVisPlots+1)
				plotType = str(paramDict[keyName])
				# TODO: should honestly just fail for incorrect input
				assert (plotType in ["field", "probe", "residual"]), (keyName+" must be either \"field\", \"probe\", or \"residual\"")
				self.numVisPlots += 1
			except:
				plotCount = False

		if (self.numVisPlots == 0):
			print("WARNING: No visualization plots selected...")
			sleep(1.0)
		self.visList = [None] * self.numVisPlots

		# initialize each figure object
		for visIdx in range(1, self.numVisPlots+1):
			
			visType = str(paramDict["visType"+str(visIdx)])

			if (visType == "field"):
				self.visList[visIdx-1] = fieldPlot(visIdx, self.visInterval, solver)
			elif (visType == "probe"):
				self.visList[visIdx-1] = probePlot(visIdx, solver)
			elif (visType == "residual"):
				assert (solver.timeIntegrator.timeType == "implicit"), "Residual visualization is only available for implicit time integrators"
				self.visList[visIdx-1] = residualPlot()
			else:
				raise ValueError("Invalid visualization selection: "+visType)

		# set plot positions/dimensions
		if (self.niceVis and self.visShow):
			try:
				self.movePlots()
			except:
				for vis in self.visList:
					vis.fig, vis.ax = plt.subplots(nrows=vis.numRows, ncols=vis.numCols, num=vis.visID, figsize=(figWidthDefault,figHeightDefault))
		else:
			for vis in self.visList:
				vis.fig, vis.ax = plt.subplots(nrows=vis.numRows, ncols=vis.numCols, num=vis.visID, figsize=(figWidthDefault,figHeightDefault))	
			
		

	def drawPlots(self, solDomain, solver):
		""" 
		Helper function to draw, display, and save all plots
		"""

		if (self.numVisPlots > 0):
			if ((solver.timeIntegrator.iter % self.visInterval) != 0):
				return

			for vis in self.visList:
				vis.plot(solDomain, solver)
				if self.visSave: vis.save(solver)

			if self.visShow:
				plt.show(block=False)
				plt.pause(0.001)

	# def savePlots(self, solver):
	# 	"""
	# 	Helper function to save displays for all plots
	# 	"""

	def movePlots(self):
		"""
		Resizes and moves plots to positions in the window for better viewing
		"""

		backend = matplotlib.get_backend()

		# check some options, do some math for plot placement
		# TODO: this doesn't account for taskbar, overestimates the size of available screen size
		fig = plt.figure(num=0)
		dpi = fig.dpi
		manager = plt.get_current_fig_manager()
		if (backend == "TkAgg"):
			manager.resize(*manager.window.maxsize())
			plt.pause(0.01)
			screenW = manager.window.winfo_width()
			screenH = manager.window.winfo_height()
		elif (backend == "WXAgg"):
			import wx
			screenW, screenH = wx.Display(0).GetGeometry().GetSize()
		elif (backend in ["Qt4Agg","Qt5Agg"]):
			# TODO: this is super sketchy and may give weird results
			manager.full_screen_toggle()
			plt.pause(0.01)
			screenW = manager.canvas.width()
			screenH = manager.canvas.height()
		else:
			raise ValueError("Nice plot positioning not supported for matplotlib backend "+backend)
		plt.close(0)

		self.numSubplotsArr = np.zeros(self.numVisPlots, dtype=np.int32)
		for visIdx, vis in enumerate(self.visList):
			self.numSubplotsArr[visIdx] = vis.numSubplots

		xBase = 50
		yBase = 50
		screenW -= xBase
		screenH -= yBase

		figX = [None] * self.numVisPlots
		figY = [None] * self.numVisPlots
		figW = [None] * self.numVisPlots
		figH = [None] * self.numVisPlots
		if (self.numVisPlots == 1):
			# just fill the screen
			figX[0], figY[0] = 0.0, 0.0
			figW[0] = screenW
			figH[0] = screenH
		elif (self.numVisPlots == 2):
			for figIdx in range(2):
				figX[figIdx] = (figIdx / 2.0) * screenW
				figY[figIdx] = 0.0
				figW[figIdx] = screenW / 2.0
				figH[figIdx] = screenH
		elif (self.numVisPlots == 3):
			allEqual = np.all(self.numSubplotsArr == self.numSubplotsArr[0])
			if allEqual:
				for figIdx in range(3):
					figW[figIdx] = screenW / 2.0
					figH[figIdx] = screenH / 2.0
					if (figIdx < 2):
						figX[figIdx] = (figIdx / 2.0) * screenW
						figY[figIdx] = 0.0
					else:
						figX[figIdx] = 0.0
						figY[figIdx] = screenH / 2.0

			else:
				largestPlot = np.argmax(self.numSubplotsArr)
				smallCounter = 0
				for figIdx in range(3):
					figW[figIdx] = screenW / 2.0
					if (figIdx == largestPlot):
						figH[figIdx] = screenH
						figX[figIdx] = 0.0
						figY[figIdx] = 0.0
					else:
						figH[figIdx] = screenH / 2.0
						figY[figIdx] = (smallCounter / 2.0) * screenH
						smallCounter += 1

		elif (self.numVisPlots == 4):
			for figIdx in range(4):
				xIdx = figIdx % 2
				yIdx = int(figIdx / 2)
				figH[figIdx] = screenH / 2.0
				figW[figIdx] = screenW / 2.0

		else:
			raise ValueError("Nice plot position not supported for more than four plots")

		# convert pixels to inches
		figW = [x/dpi for x in figW]
		figH = [x/dpi for x in figH]

		# move plots
		for visIdx, vis in enumerate(self.visList):

			vis.fig, vis.ax = plt.subplots(nrows=vis.numRows, ncols=vis.numCols, num=vis.visID, figsize=(figW[visIdx],figH[visIdx]))
			window = vis.fig.canvas.manager.window
			if (backend == "TkAgg"):
				window.wm_geometry("+%d+%d" % (figX[visIdx], figY[visIdx]))
			elif (backend == "WXAgg"):
				window.SetPosition((figX[visIdx], figY[visIdx]))
			elif (backend in ["Qt4Agg","Qt5Agg"]):
				window.move(figX[visIdx], figY[visIdx])
			
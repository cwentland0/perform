from pygems1d.visualization.fieldPlot import fieldPlot
# from pygems1d.visualization.probePlot import probePlot
# from pygems1d.visualization.residualPlot import residualPlot
from pygems1d.inputFuncs import catchInput

import pdb
import matplotlib.pyplot as plt
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

		# count number of visualizations requested
		self.numVisPlots = 0
		plotCount = True
		while plotCount:
			try:
				keyName = "visType"+str(self.numVisPlots+1)
				plotType = str(paramDict[keyName])
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
				self.visList[visIdx-1] = probePlot()
			elif (visType == "residual"):
				assert (solver.timeIntegrator.timeType == "implicit"), "Residual visualization is only available for implicit time integrators"
				self.visList[visIdx-1] = residualPlot()
			else:
				raise ValueError("Invalid visualization selection: "+visType)


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


	def updatePointData(self):
		"""
		Append data to point plots in visualization group
		"""

		pass

	def updateResidualData(self):
		"""
		Append data to residual plots in visualization group
		"""

		pass
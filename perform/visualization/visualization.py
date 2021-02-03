from perform.inputFuncs import catchList

import numpy as np
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
mpl.rc('font', family='serif',size='8') 	# TODO: adapt axis label font size to number of subplots
mpl.rc('axes', labelsize='x-large')
mpl.rc('figure', facecolor='w')
mpl.rc('text', usetex=False)
mpl.rc('text.latex',preamble=r'\usepackage{amsmath}')

# TODO: this is kind of a mess
# 	I'm struggling to write effective hierarchy for field, probe, and residual, as probe shares similarities with field and residual plots
# TODO: add RHS, flux plotting

class visualization:

	def __init__(self, solDomain, solver):

		paramDict = solver.paramDict

		if (self.visType in ["field","probe"]):

			# check requested variables
			self.visVar	= catchList(paramDict, "visVar"+str(self.visID), [None])
			for visVar in self.visVar:
				if (visVar in ["pressure","velocity","temperature","source","density","momentum","energy"]):
					pass
				elif ((visVar[:7] == "species") or (visVar[:15] == "density-species")):
					try:
						if (visVar[:7] == "species"):
							speciesIdx = int(visVar[7:])
						elif (visVar[:15] == "density-species"):
							speciesIdx = int(visVar[15:])

						assert ((speciesIdx > 0) and (speciesIdx <= solDomain.gasModel.numSpeciesFull)), \
							"Species number must be a positive integer less than or equal to the number of chemical species"
					except:
						raise ValueError("visVar entry " + visVar + " must be formated as speciesX or density-speciesX, where X is an integer")
				else:
					raise ValueError("Invalid entry in visVar"+str(visID))

			self.numSubplots = len(self.visVar)

		# residual plot
		else:
			self.visVar = ["residual"]
			self.numSubplots = 1

		self.visXBounds 	= catchList(paramDict, "visXBounds"+str(self.visID), [[None,None]], lenHighest=self.numSubplots)
		self.visYBounds 	= catchList(paramDict, "visYBounds"+str(self.visID), [[None,None]], lenHighest=self.numSubplots)

		if (self.numSubplots == 1):
			self.numRows = 1 
			self.numCols = 1
		elif (self.numSubplots == 2):
			self.numRows = 2
			self.numCols = 1
		elif (self.numSubplots <= 4):
			self.numRows = 2
			self.numCols = 2
		elif (self.numSubplots <= 6):
			self.numRows = 3
			self.numCols = 2
		elif (self.numSubplots <= 9):
			# TODO: an extra, empty subplot shows up with 7 subplots
			self.numRows = 3
			self.numCols = 3
		else:
			raise ValueError("Cannot plot more than nine subplots in the same image")

		# axis labels
		# TODO: could change this to a dictionary reference
		self.axLabels = [None] * self.numSubplots
		if (self.visType == "residual"):
			self.axLabels[0] = "Residual History"
		else:
			for axIdx in range(self.numSubplots):
				varStr = self.visVar[axIdx]
				if (varStr == "pressure"):
					self.axLabels[axIdx] = "Pressure (Pa)"
				elif (varStr == "velocity"):
					self.axLabels[axIdx] = "Velocity (m/s)"
				elif (varStr == "temperature"):
					self.axLabels[axIdx] = "Temperature (K)"
				elif (varStr == "source"):
					self.axLabels[axIdx] = "Reaction Source Term"
				elif (varStr == "density"):
					self.axLabels[axIdx] = "Density (kg/m^3)"
				elif (varStr == "momentum"):
					self.axLabels[axIdx] = "Momentum (kg/s-m^2)"
				elif (varStr == "energy"):
					self.axLabels[axIdx] = "Total Energy"

				# TODO: some way to incorporate actual species name
				elif (varStr[:7] == "species"):
					self.axLabels[axIdx] = "Species "+str(varStr[7:])+" Mass Fraction" 
				elif (varStr[:15] == "density-species"):
					self.axLabels[axIdx] = "Density-weighted Species "+str(varStr[7:])+" Mass Fraction (kg/m^3)"
				else:
					raise ValueError("Invalid field visualization variable:"+str(solver.visVar))


	def plot(self, solDomain, solver):
		"""
		Draw and display plot
		"""

		plt.figure(self.visID)

		if (type(self.ax) != np.ndarray):
			axList = [self.ax]
		else:
			axList = self.ax

		for colIdx, col in enumerate(axList):
			if (type(col) != np.ndarray):
				colList = [col]
			else:
				colList = col
			for rowIdx, axVar in enumerate(colList):

				axVar.cla()
				linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (self.numRows, self.numCols))[0]
				if ((linIdx+1) > self.numSubplots): 
					axVar.axis("off")
					break

				yData = self.getYData(solDomain, self.visVar[linIdx], solver)
				xData = self.getXData(solDomain, solver)

				axVar.plot(xData, yData)
				axVar.set_ylim(self.visYBounds[linIdx])
				axVar.set_xlim(self.visXBounds[linIdx])
				axVar.set_ylabel(self.axLabels[linIdx])
				axVar.set_xlabel(self.xLabel)
				
				if (self.visType == "field"):
					axVar.ticklabel_format(useOffset=False)
				else:
					axVar.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

		self.fig.tight_layout()
		self.fig.canvas.draw_idle()
import perform.constants as const
from perform.visualization.visualization import visualization
from perform.inputFuncs import catchInput

import os
import matplotlib.pyplot as plt
import numpy as np
import pdb

# TODO: might be easier to make probe and residual plots under some pointPlot class
# TODO: move some of the init input arguments used for assertions outside

class probePlot(visualization):

	def __init__(self, visID, simType, probeVars, visVars, probeNum, numProbes, visXBounds, visYBounds, numSpeciesFull):

		self.visType = "probe"
		self.visID   = visID
		self.xLabel  = "t (s)"

		self.probeNum  = probeNum
		self.probeVars = probeVars
		assert (self.probeNum >= 0 ), "Must provide positive integer probe number for probe"+str(self.visID)
		assert (self.probeNum < numProbes), "probeNum"+str(self.visID)+" must correspond to a valid probe"

		super().__init__(visID, visVars, visXBounds, visYBounds, numSpeciesFull)

		# image file on disk
		visName = ""
		for visVar in self.visVars:
			visName += "_"+visVar
		figName = "probe" + visName + "_" + str(self.probeNum) + "_" + simType + ".png"
		self.figFile = os.path.join(const.imageOutputDir, figName) 

		# check that requested variables are being probed
		for visVar in self.visVars:
			assert (visVar in probeVars), "Must probe "+visVar+" to plot it"


	def plot(self, probeVals, timeVals, iterNum, lineStyle):
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

				linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (self.numRows, self.numCols))[0]
				if ((linIdx+1) > self.numSubplots): 
					axVar.axis("off")
					break

				yData = self.getYData(probeVals, self.visVars[linIdx], iterNum)
				xData = timeVals[:iterNum]

				axVar.plot(xData, yData, lineStyle)
				axVar.set_ylim(self.visYBounds[linIdx])
				axVar.set_xlim(self.visXBounds[linIdx])
				axVar.set_ylabel(self.axLabels[linIdx])
				axVar.set_xlabel(self.xLabel)
				
				axVar.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

		self.fig.tight_layout()
		self.fig.canvas.draw_idle()


	def getYData(self, probeVals, varStr, iterNum):

		# data extraction of probes is done in solDomain 
		varIdx = np.squeeze(np.argwhere(self.probeVars == varStr)[0])
		yData = probeVals[self.probeNum, varIdx, :iterNum]

		return yData


	def save(self, iterNum):

		plt.figure(self.visID)
		self.fig.savefig(self.figFile)

		
import perform.constants as const
from perform.visualization.visualization import visualization

import numpy as np
import matplotlib.pyplot as plt
import os
from math import floor, log

class fieldPlot(visualization):
	"""
	Class for field plot image
	"""

	def __init__(self, visID, visInterval, numSteps, simType, visVars, visXBounds, visYBounds, numSpeciesFull):

		self.visType 		= "field"
		self.visInterval	= visInterval
		self.visID 			= visID
		self.xLabel 		= "x (m)"

		self.numImgs 		= int(numSteps / visInterval)
		if (self.numImgs > 0):
			self.imgString 	= '%0'+str(floor(log(self.numImgs, 10))+1)+'d'
		else:
			self.imgString 	= None

		super().__init__(visID, visVars, visXBounds, visYBounds, numSpeciesFull)

		# set up output directory
		visName = ""
		for visVar in self.visVars:
			visName += "_" + visVar
		visName += "_" + simType
		self.imgDir = os.path.join(const.imageOutputDir, "field"+visName)
		if not os.path.isdir(self.imgDir): os.mkdir(self.imgDir)
		

	def plot(self, solPrim, solCons, source, rhs, gasModel, xCell, lineStyle):
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

				yData = self.getYData(solPrim, solCons, source, rhs, self.visVars[linIdx], gasModel)
				xData = xCell

				axVar.plot(xData, yData, lineStyle)
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


	def getYData(self, solPrim, solCons, source, rhs, varStr, gasModel):
		"""
		Extract plotting data from flow field domain data
		"""

		try:
			if (varStr == "pressure"):
				yData = solPrim[0,:]
			elif (varStr == "velocity"):
				yData = solPrim[1,:]
			elif (varStr == "temperature"):
				yData = solPrim[2,:]
			elif (varStr == "source"):
				yData = source[0,:]
			elif (varStr == "density"):
				yData = solCons[0,:]
			elif (varStr == "momentum"):
				yData = solCons[1,:]
			elif (varStr == "energy"):
				yData = solCons[2,:]
			elif (varStr[:7] == "species"):
				specIdx = int(varStr[7:])
				if (specIdx == gasModel.numSpeciesFull):
					massFracs = gasModel.calcAllMassFracs(solPrim[3:,:], threshold=False)
					yData = massFracs[-1,:]
				else:
					yData = solPrim[3+specIdx-1,:]
			elif (varStr[:15] == "density-species"):
				specIdx = int(varStr[15:])
				yData = solCons[3+specIdx-1,:]
		except Exception as e:
			print(e)
			raise ValueError("Invalid field visualization variable:"+str(varStr))

		return yData


	def save(self, iterNum):
		"""
		Save plot to disk
		"""

		plt.figure(self.visID)
		visIdx 	= int(iterNum / self.visInterval)
		figNum 	= self.imgString % visIdx
		figFile = os.path.join(self.imgDir, "fig_"+figNum+".png")
		self.fig.savefig(figFile)
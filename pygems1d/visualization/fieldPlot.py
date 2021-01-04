import pygems1d.constants as const
from pygems1d.visualization.visualization import visualization
from pygems1d.inputFuncs import catchList

import os
from math import floor, log
import numpy as np
import pdb

class fieldPlot(visualization):
	"""
	Class for field plot image
	"""

	def __init__(self, plotID, visInterval, solver):

		paramDict = solver.paramDict

		self.visType 		= "field"
		self.visInterval	= visInterval
		
		# check requested variables
		self.visVar			= catchList(paramDict, "visVar"+str(plotID), [None])
		for visVar in self.visVar:
			if (visVar in ["pressure","velocity","temperature","source","density","momentum","energy"]):
				pass
			elif ((visVar[:7] == "species") or (visVar[:15] == "density-species")):
				try:
					if (visVar[:7] == "species"):
						speciesIdx = int(visVar[7:])
					elif (visVar[:15] == "density-species"):
						speciesIdx = int(visVar[15:])

					assert ((speciesIdx > 0) and (speciesIdx <= solver.gasModel.numSpecies)), \
						"Species number must be a positive integer less than or equal to the number of chemical species"
				except:
					raise ValueError("visVar entry" + visVar + " must be formated as speciesX or density-speciesX, where X is an integer")
			else:
				raise ValueError("Invalid entry in visVar"+str(plotID))


		self.numSubplots 	= len(self.visVar)
		self.visXBounds 	= catchList(paramDict, "visXBounds"+str(plotID), [[None,None]], lenHighest=self.numSubplots)
		self.visYBounds 	= catchList(paramDict, "visYBounds"+str(plotID), [[None,None]], lenHighest=self.numSubplots)

		self.numImgs 		= int(solver.timeIntegrator.numSteps / visInterval)
		if (self.numImgs > 0):
			self.imgString 	= '%0'+str(floor(log(self.numImgs, 10))+1)+'d'
		else:
			self.imgString 	= None

		super().__init__(self.visType, solver)

		# set up output directory
		visName = ""
		for visVar in self.visVar:
			visName += "_"+visVar
		visName += "_"+solver.simType
		self.imgDir = os.path.join(const.imageOutputDir, "field"+visName)
		if not os.path.isdir(self.imgDir): os.mkdir(self.imgDir)
		

	def plot(self, solDomain, solver):
		"""
		Draw and display field line plot(s)
		"""

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

				varStr = self.visVar[linIdx]
				field = self.getFieldData(solDomain, varStr)

				axVar.plot(solver.mesh.xCell, field)
				axVar.set_ylim(self.visYBounds[linIdx])
				axVar.set_xlim(self.visXBounds[linIdx])
				axVar.set_ylabel(self.axLabels[linIdx])
				axVar.set_xlabel("x (m)")
				axVar.ticklabel_format(useOffset=False)

		self.fig.tight_layout()

	def save(self, solver):
		"""
		Save plot to disk
		"""

		visIdx 	= int(solver.timeIntegrator.iter / self.visInterval)
		figNum 	= self.imgString % visIdx
		figFile = os.path.join(self.imgDir, "fig_"+figNum+".png")
		fig.savefig(figFile)
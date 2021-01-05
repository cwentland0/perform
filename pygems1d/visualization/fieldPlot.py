import pygems1d.constants as const
from pygems1d.visualization.visualization import visualization
from pygems1d.inputFuncs import catchList

import os
from math import floor, log
import numpy as np
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
mpl.rc('font', family='serif',size='8')
mpl.rc('axes', labelsize='x-large')
mpl.rc('figure', facecolor='w')
mpl.rc('text', usetex=False)
mpl.rc('text.latex',preamble=r'\usepackage{amsmath}')

class fieldPlot(visualization):
	"""
	Class for field plot image
	"""

	def __init__(self, visID, visInterval, solver):

		paramDict = solver.paramDict

		self.visType 		= "field"
		self.visInterval	= visInterval
		self.visID 			= visID
		
		# check requested variables
		self.visVar			= catchList(paramDict, "visVar"+str(visID), [None])
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
				raise ValueError("Invalid entry in visVar"+str(visID))


		self.numSubplots 	= len(self.visVar)
		self.visXBounds 	= catchList(paramDict, "visXBounds"+str(visID), [[None,None]], lenHighest=self.numSubplots)
		self.visYBounds 	= catchList(paramDict, "visYBounds"+str(visID), [[None,None]], lenHighest=self.numSubplots)

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

				varStr = self.visVar[linIdx]
				field = self.getFieldData(solDomain, varStr)

				axVar.plot(solver.mesh.xCell, field)
				axVar.set_ylim(self.visYBounds[linIdx])
				axVar.set_xlim(self.visXBounds[linIdx])
				axVar.set_ylabel(self.axLabels[linIdx])
				axVar.set_xlabel("x (m)")
				axVar.ticklabel_format(useOffset=False)

		self.fig.tight_layout()
		self.fig.canvas.draw_idle()

	def getFieldData(self, solDomain, varStr):
		"""
		Extract plotting data from flow field domain data
		"""

		solPrim = solDomain.solInt.solPrim
		solCons = solDomain.solInt.solCons
		source  = solDomain.solInt.source
		rhs 	= solDomain.solInt.RHS

		try:
			if (varStr == "pressure"):
				extrData = solPrim[:,0]
			elif (varStr == "velocity"):
				extrData = solPrim[:,1]
			elif (varStr == "temperature"):
				extrData = solPrim[:,2]
			elif (varStr == "source"):
				extrData = source[:,0]
			elif (varStr == "density"):
				extrData = solCons[:,0]
			elif (varStr == "momentum"):
				extrData = solCons[:,1]
			elif (varStr == "energy"):
				extrData = solCons[:,2]

			# TODO: fails for last species
			elif (varStr[:7] == "species"):
				specIdx = int(varStr[7:])
				extrData = solPrim[:,3+specIdx-1]
			elif (varStr[:15] == "density-species"):
				specIdx = int(varStr[15:])
				extrData = solCons[:,3+specIdx-1]
		except:
			raise ValueError("Invalid field visualization variable:"+str(varStr))

		return extrData

	def save(self, solver):
		"""
		Save plot to disk
		"""

		visIdx 	= int(solver.timeIntegrator.iter / self.visInterval)
		figNum 	= self.imgString % visIdx
		figFile = os.path.join(self.imgDir, "fig_"+figNum+".png")
		fig.savefig(figFile)
import perform.constants as const
from perform.visualization.visualization import visualization
from perform.inputFuncs import catchInput

import os
import matplotlib.pyplot as plt
import numpy as np
import pdb

# TODO: might be easier to make probe and residual plots under some pointPlot class

class probePlot(visualization):

	def __init__(self, visID, solDomain, solver):

		self.visType = "probe"
		self.visID   = visID
		self.xLabel  = "t (s)"

		self.probeNum = catchInput(solver.paramDict, "probeNum"+str(self.visID), -2) - 1
		assert (self.probeNum >= 0 ), "Must provide positive integer probe number for probe"+str(self.visID)
		assert (self.probeNum < solver.numProbes), "probeNum"+str(self.visID)+" must correspond to a valid probe"

		super().__init__(solDomain, solver)

		# image file on disk
		visName = ""
		for visVar in self.visVar:
			visName += "_"+visVar
		figName = "probe" + visName + "_" + str(self.probeNum) + "_" + solver.simType + ".png"
		self.figFile = os.path.join(const.imageOutputDir, figName) 

		# check that requested variables are being probed
		for visVar in self.visVar:
			assert (visVar in solver.probeVars), "Must probe "+visVar+" to plot it"

	def getYData(self, solDomain, varStr, solver):

		# data extraction of probes is done in solDomain 
		varIdx = np.squeeze(np.argwhere(solDomain.probeVars == varStr)[0])
		yData = solDomain.probeVals[self.probeNum, varIdx, :solver.iter]

		return yData

	def getXData(self, solDomain, solver):

		xData = solDomain.timeVals[:solver.iter]
		return xData

	def save(self, solver):

		plt.figure(self.visID)
		self.fig.savefig(self.figFile)

		
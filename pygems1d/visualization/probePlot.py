from pygems1d.visualization.visualization import visualization
from pygems1d.inputFuncs import catchInput

import numpy as np
import pdb

# TODO: might be easier to make probe and residual plots under some pointPlot class

class probePlot(visualization):

	def __init__(self, visID, solver):

		self.visType = "probe"
		self.visID   = visID
		self.xLabel  = "t (s)"

		self.probeNum = catchInput(solver.paramDict, "probeNum"+str(self.visID), -2) - 1
		assert (self.probeNum >= 0 ), "Must provide positive integer probe number for probe"+str(self.visID)
		assert (self.probeNum < solver.numProbes), "probeNum"+str(self.visID)+" must correspond to a valid probe"

		super().__init__(solver)

		# check that requested variables are being probed
		for visVar in self.visVar:
			assert (visVar in solver.probeVars), "Must probe "+visVar+" to plot it"

	# TODO: can generalize this for visualization object, just make a getYVals and getXVals method in each plot class
	# def plot(self):

	# 	plt.figure(self.visID)

	# 	if (type(self.ax) != np.ndarray): 
	# 		axList = [self.ax]
	# 	else:
	# 		axList = self.ax 

	# 	for colIdx, col in enumerate(axList):
	# 		if (type(col) != np.ndarray):
	# 			colList = [col]
	# 		else:
	# 			colList = col
	# 		for rowIdx, axVar in enumerate(colList):

	# 			axVar.cla()
	# 			linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (self.numRows, solver.visNCols))[0]
	# 			if ((linIdx+1) > solver.numVis): 
	# 				axVar.axis("off")
	# 				break

	# 			axVar.plot(tVals[:solver.timeIntegrator.iter], probeVals[:solver.timeIntegrator.iter, linIdx])
	# 			axVar.set_ylim(solver.visYBounds[linIdx])
	# 			axVar.set_xlim(solver.visXBounds[linIdx])
	# 			axVar.set_ylabel(axLabels[linIdx])
	# 			axVar.set_xlabel()
	# 			axVar.ticklabel_format(axis='both',useOffset=False)
	# 			

	# 	fig.tight_layout()
	# 	plt.show(block=False)
	# 	plt.pause(0.001)

	def getYData(self, solDomain, varStr, solver):

		# data extraction of probes is done in solDomain 
		varIdx = np.squeeze(np.argwhere(solDomain.probeVars == varStr)[0])
		yData = solDomain.probeVals[self.probeNum, varIdx, :solver.timeIntegrator.iter]

		return yData

	def getXData(self, solDomain, solver):

		xData = solDomain.timeVals[:solver.timeIntegrator.iter]
		return xData


	

		
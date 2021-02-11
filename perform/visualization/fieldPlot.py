import perform.constants as const
from perform.visualization.visualization import visualization

import matplotlib.pyplot as plt
import os
from math import floor, log

class fieldPlot(visualization):
	"""
	Class for field plot image
	"""

	def __init__(self, visID, visInterval, solDomain, solver):

		paramDict = solver.paramDict

		self.visType 		= "field"
		self.visInterval	= visInterval
		self.visID 			= visID
		self.xLabel 		= "x (m)"

		self.numImgs 		= int(solver.numSteps / visInterval)
		if (self.numImgs > 0):
			self.imgString 	= '%0'+str(floor(log(self.numImgs, 10))+1)+'d'
		else:
			self.imgString 	= None

		super().__init__(solDomain, solver)

		# set up output directory
		visName = ""
		for visVar in self.visVar:
			visName += "_"+visVar
		visName += "_"+solver.simType
		self.imgDir = os.path.join(const.imageOutputDir, "field"+visName)
		if not os.path.isdir(self.imgDir): os.mkdir(self.imgDir)
		

	def getYData(self, solDomain, varStr, solver):
		"""
		Extract plotting data from flow field domain data
		"""

		solPrim = solDomain.solInt.solPrim
		solCons = solDomain.solInt.solCons
		source  = solDomain.solInt.source
		rhs 	= solDomain.solInt.RHS

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
				if (specIdx == solDomain.gasModel.numSpeciesFull):
					massFracs = solDomain.gasModel.calcAllMassFracs(solPrim[3:,:])
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

	def getXData(self, solDomain, solver):
		
		xData = solver.mesh.xCell
		return xData

	def save(self, solver):
		"""
		Save plot to disk
		"""

		plt.figure(self.visID)
		visIdx 	= int(solver.iter / self.visInterval)
		figNum 	= self.imgString % visIdx
		figFile = os.path.join(self.imgDir, "fig_"+figNum+".png")
		self.fig.savefig(figFile)
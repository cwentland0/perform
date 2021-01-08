
from pygems1d.inputFuncs import readInputFile, catchList

from time import sleep
import pdb

class romDomain:
	"""
	Container class for ROM parameters and romModels
	"""

	def __init__(self, solver):

		romDict = readInputFile(solver.romInputs)
		self.romDict = romDict

		# catch method and latent dimensions for each model
		self.numModels = int(romDict["numModels"])
		self.romMethods = catchList(romDict, "romMethods", [""])
		self.latentDims = catchList(romDict, "latentDims", [0])

		for i in self.latentDims: assert (i > 0), "latentDims must contain positive integers"
		if (self.numModels == 1):
			assert (len(self.romMethods) == 1), "Must provide only one value of romMethods when numModels = 1"
			assert (len(self.latentDims) == 1), "Must provide only one value of latentDims when numModels = 1"
			assert (self.latentDims[0] > 0), "latentDims must contain positive integers"
		else:
			if (len(self.romMethods) == self.numModels):
				pass
			elif (len(self.romMethods) == 1):
				print("Only one value provided in romMethods, applying to all models")
				sleep(1.0)
				self.romMethods = [self.romMethods[0]] * self.numModels
			else:
				raise ValueError("Must provide either numModels or 1 entry in romMethods")

			if (len(self.latentDims) == self.numModels):
				pass
			elif (len(self.latentDims) == 1):	
				print("Only one value provided in latentDims, applying to all models")
				sleep(1.0)
				self.latentDims = [self.latentDims[0]] * self.numModels
			else:
				raise ValueError("Must provide either numModels or 1 entry in latentDims")
				
	
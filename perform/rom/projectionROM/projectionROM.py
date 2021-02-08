from perform.rom.romModel import romModel
from perform.constants import realType

import numpy as np

class projectionROM(romModel):
	"""
	Base class for projection-based reduced-order model
	This model makes no assumption on the form of the decoder, 
		but assumes a linear projection onto the low-dimensional space
	"""

	def __init__(self, modelIdx, romDomain, solver):

		super().__init__(modelIdx, romDomain, solver)


	def projectToLowDim(self, projector, fullDimArr, transpose=False):
		"""
		Project given full-dimensional vector onto low-dimensional space via given projector
		Assumed that fullDimArr is either 1D array or is in [numVars, numCells] order
		Assumed that projector is already in [numModes, numVars x numCells] order
		"""
		
		if (fullDimArr.ndim == 2):
			fullDimVec = fullDimArr.flatten(order="C")
		elif (fullDimArr.ndim == 1):
			fullDimVec = fullDimArr.copy()
		else:
			raise ValueError("fullDimArr must be one- or two-dimensional")

		if transpose:
			codeOut = projector.T @ fullDimVec
		else:
			codeOut = projector @ fullDimVec

		return codeOut 


	def calcRHSLowDim(self, romDomain, solDomain):
		"""
		Project RHS onto low-dimensional space
		Assumes that RHS term is scaled using an appropriate conservative variable normalization profile
		"""

		# scale RHS
		normSubProf = np.zeros(self.normFacProfCons.shape, dtype=realType)
		rhsScaled = self.standardizeData(solDomain.solInt.RHS[self.varIdxs[:,None], solDomain.directSampIdxs[None,:]], 
										 normalize=True, normFacProf=self.normFacProfCons[:,solDomain.directSampIdxs], normSubProf=normSubProf[:,solDomain.directSampIdxs],
										 center=False, inverse=False)

		# calc projection operator and project
		self.calcProjector(solDomain, False)
		pdb.set_trace()
		self.rhsLowDim = self.projectToLowDim(self.projector, rhsScaled, transpose=False)
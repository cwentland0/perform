from pygems1d.constants import realType
from pygems1d.rom.romModel import romModel
from pygems1d.inputFuncs import catchInput

import numpy as np
import pdb

class linearProjROM(romModel):

	def __init__(self, modelIdx, romDomain, solDomain, romDict, solver):


		self.gappyPOD = catchInput(romDict, "gappyPOD", False)

		super().__init__(modelIdx, romDomain, solDomain, romDict, solver)

		# load and check trial basis
		self.trialBasis = np.load(romDomain.modelFiles[self.modelIdx])
		numVarsBasisIn, numCellsBasisIn, numModesBasisIn = self.trialBasis.shape
		assert (numVarsBasisIn == self.numVars), ("Basis at " + romDomain.modelFiles[self.modelIdx] + " represents a different number of variables " +
			"than specified by modelVarIdxs (" + str(numVarsBasisIn) + " != " + str(self.numVars) + ")")
		assert (numCellsBasisIn == solver.mesh.numCells), ("Basis at " + romDomain.modelFiles[self.modelIdx] + " has a different number of cells " +
			"than the physical domain (" + str(numCellsBasisIn) + " != " + str(solver.mesh.numCells) + ")")
		assert (numModesBasisIn >= self.latentDim), ("Basis at " + romDomain.modelFiles[self.modelIdx] + " must have at least " + str(self.latentDim) +
			" modes (" + str(numModesBasisIn) + " < " + str(self.latentDim) + ")")

		self.trialBasis = self.trialBasis[:,:,:self.latentDim]
		self.trialBasis = np.reshape(self.trialBasis, (-1, self.latentDim), order='C')

		# load and check gappy POD parameters and basis, if requested
		# TODO: do this
	

	def applyTrialBasis(self, code):
		"""
		Compute raw decoding of code, without de-normalizing or de-centering
		"""

		sol = self.trialBasis @ code
		sol = np.reshape(sol, (self.numVars, -1), order="C")
		return sol


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


	def calcRHSLowDim(self, solDomain):
		"""
		Project RHS onto low-dimensional space
		"""

		# scale RHS
		# NOTE: normalization subtractive factor is not applied here
		# TODO: could modify standardizeData() to not require the subtractive element
		normSubProf = np.zeros(self.normFacProfCons.shape, dtype=realType)
		rhsScaled = self.standardizeData(solDomain.solInt.RHS[self.varIdxs, :], 
										 normalize=True, normFacProf=self.normFacProfCons, normSubProf=normSubProf,
										 center=False, inverse=False)

		# calc projection operator and project
		self.calcProjector()
		self.rhsLowDim = self.projectToLowDim(self.projector, rhsScaled, transpose=False)



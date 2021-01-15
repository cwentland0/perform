from pygems1d.constants import realType
from pygems1d.rom.romModel import romModel
from pygems1d.inputFuncs import catchInput

import numpy as np
import pdb

class linearProjROM(romModel):

	def __init__(self, modelIdx, romDomain, solver):

		super().__init__(modelIdx, romDomain, solver)

		# load and check trial basis
		self.trialBasis = np.load(romDomain.modelFiles[self.modelIdx])
		numVarsBasisIn, numCellsBasisIn, numModesBasisIn = self.trialBasis.shape
		assert (numVarsBasisIn == self.numVars), ("Basis at " + romDomain.modelFiles[self.modelIdx] + " represents a different number of variables " +
			"than specified by modelVarIdxs (" + str(numVarsBasisIn) + " != " + str(self.numVars) + ")")
		assert (numCellsBasisIn == solver.mesh.numCells), ("Basis at " + romDomain.modelFiles[self.modelIdx] + " has a different number of cells " +
			"than the physical domain (" + str(numCellsBasisIn) + " != " + str(solver.mesh.numCells) + ")")
		assert (numModesBasisIn >= self.latentDim), ("Basis at " + romDomain.modelFiles[self.modelIdx] + " must have at least " + str(self.latentDim) +
			" modes (" + str(numModesBasisIn) + " < " + str(self.latentDim) + ")")

		# flatten first two dimensions for easier matmul
		self.trialBasis = self.trialBasis[:,:,:self.latentDim]
		self.trialBasis = np.reshape(self.trialBasis, (-1, self.latentDim), order='C')

		# load and check gappy POD basis
		if romDomain.hyperReduc:
			hyperReducBasis = np.load(romDomain.hyperReducFiles[self.modelIdx])
			assert (hyperReducBasis.ndim == 3), "Hyper-reduction basis must have three axes"
			assert (hyperReducBasis.shape[:2] == (solver.gasModel.numEqs, solver.mesh.numCells)), \
				"Hyper reduction basis must have shape [numEqs, numCells, numHRModes]"

			self.hyperReducDim = romDomain.hyperReducDims[self.modelIdx]
			hyperReducBasis = hyperReducBasis[:,:,:self.hyperReducDim]
			self.hyperReducBasis = np.reshape(hyperReducBasis, (-1, self.hyperReducDim), order="C")

			# indices for sampling flattened hyperReducBasis
			self.directHyperReducSampIdxs = np.zeros(romDomain.numSampCells * self.numVars, dtype=np.int32)
			for varNum in range(self.numVars):
				idx1 = varNum * romDomain.numSampCells
				idx2 = (varNum+1) * romDomain.numSampCells
				self.directHyperReducSampIdxs[idx1:idx2] = romDomain.directSampIdxs + varNum * solver.mesh.numCells


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


	def calcRHSLowDim(self, romDomain, solDomain):
		"""
		Project RHS onto low-dimensional space
		"""

		# scale RHS
		normSubProf = np.zeros(self.normFacProfCons.shape, dtype=realType)
		rhsScaled = self.standardizeData(solDomain.solInt.RHS[self.varIdxs[:,None], solDomain.directSampIdxs[None,:]], 
										 normalize=True, normFacProf=self.normFacProfCons[:,solDomain.directSampIdxs], normSubProf=normSubProf[:,solDomain.directSampIdxs],
										 center=False, inverse=False)

		# calc projection operator and project
		self.calcProjector(romDomain, romDomain.adaptiveROM)
		self.rhsLowDim = self.projectToLowDim(self.projector, rhsScaled, transpose=False)

from perform.constants import realType
from perform.rom.projectionROM.projectionROM import projectionROM
from perform.inputFuncs import catchInput

import numpy as np
import pdb

class linearProjROM(projectionROM):
	"""
	Base class for all projection-based ROMs which use a linear basis representation
	"""


	def __init__(self, modelIdx, romDomain, solver, solDomain):

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
		self.trialBasis = self.trialBasis[self.varIdxs,:,:self.latentDim]
		self.trialBasis = np.reshape(self.trialBasis, (-1, self.latentDim), order='C')

		# load and check gappy POD basis
		if romDomain.hyperReduc:
			hyperReducBasis = np.load(romDomain.hyperReducFiles[self.modelIdx])
			assert (hyperReducBasis.ndim == 3), "Hyper-reduction basis must have three axes"
			assert (hyperReducBasis.shape[:2] == (solDomain.gasModel.numEqs, solver.mesh.numCells)), \
				"Hyper reduction basis must have shape [numEqs, numCells, numHRModes]"

			self.hyperReducDim = romDomain.hyperReducDims[self.modelIdx]
			hyperReducBasis = hyperReducBasis[self.varIdxs,:,:self.hyperReducDim]
			self.hyperReducBasis = np.reshape(hyperReducBasis, (-1, self.hyperReducDim), order="C")

			# indices for sampling flattened hyperReducBasis
			self.directHyperReducSampIdxs = np.zeros(romDomain.numSampCells * self.numVars, dtype=np.int32)
			for varNum in range(self.numVars):
				idx1 = varNum * romDomain.numSampCells
				idx2 = (varNum+1) * romDomain.numSampCells
				self.directHyperReducSampIdxs[idx1:idx2] = romDomain.directSampIdxs + varNum * solver.mesh.numCells


	def initFromSol(self, solDomain):
		"""
		Initialize full-order solution from projection of loaded full-order initial conditions
		"""

		if (self.targetCons):
			sol = self.standardizeData(solDomain.solInt.solCons[self.varIdxs, :], 
										normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons,
										center=True, centProf=self.centProfCons, inverse=False)
			self.code = self.projectToLowDim(self.trialBasis, sol, transpose=True)
			solDomain.solInt.solCons[self.varIdxs, :] = self.decodeSol(self.code)
		else:
			sol = self.standardizeData(solDomain.solInt.solPrim[self.varIdxs, :], 
										normalize=True, normFacProf=self.normFacProfPrim, normSubProf=self.normSubProfPrim,
										center=True, centProf=self.centProfPrim, inverse=False)
			self.code = self.projectToLowDim(self.trialBasis, sol, transpose=True)
			solDomain.solInt.solPrim[self.varIdxs, :] = self.decodeSol(self.code)


	def applyDecoder(self, code):
		"""
		Compute raw decoding of code, without de-normalizing or de-centering
		"""

		sol = self.trialBasis @ code
		sol = np.reshape(sol, (self.numVars, -1), order="C")
		return sol

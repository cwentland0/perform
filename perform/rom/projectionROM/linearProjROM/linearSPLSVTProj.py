from perform.constants import realType
from perform.rom.linearProjROM.linearProjROM import linearProjROM

import numpy as np
import pdb

class linearSPLSVTProj(linearProjROM):
	"""
	Class for linear decoder and Galerkin projection
	Trial basis is assumed to represent the primitive variables (see Galerkin/LSPG for primitive variable representation)
	"""

	def __init__(self, modelIdx, romDomain, solver):

		super().__init__(modelIdx, romDomain, solver)

		trialBasisF = np.reshape(self.trialBasis, (self.numVars, solver.mesh.numCells, self.latentDim), order='C')
		self.trialBasisF = np.reshape(trialBasisF, (-1, self.latentDim), order="F")
		self.trialBasisFScaled = self.normFacProfPrim.ravel(order="F")[:,None] * self.trialBasisF
		self.testBasis = np.zeros(self.trialBasis.shape, dtype=realType)
		

	def decodeSol(self, code):
		"""
		Compute full decoding of primitive solution, including decentering and denormalization
		"""

		solPrim = self.applyTrialBasis(code)
		solPrim = self.standardizeData(solPrim, 
									   normalize=True, normFacProf=self.normFacProfPrim, normSubProf=self.normSubProfPrim,
									   center=True, centProf=self.centProfPrim, inverse=True)
		return solPrim

	def initFromCode(self, code0, solDomain, solver):
		"""
		Initialize full-order primitive solution from input low-dimensional state
		"""

		self.code = code0.copy()
		solDomain.solInt.solPrim[self.varIdxs, :] = self.decodeSol(self.code)


	def initFromSol(self, solDomain, solver):
		"""
		Initialize full-order primitive solution from projection of loaded full-order initial conditions
		"""

		solPrim = self.standardizeData(solDomain.solInt.solPrim[self.varIdxs, :], 
									   normalize=True, normFacProf=self.normFacProfPrim, normSubProf=self.normSubProfPrim, 
									   center=True, centProf=self.centProfPrim, inverse=False)
		self.code = self.projectToLowDim(self.trialBasis, solPrim, transpose=True)
		solDomain.solInt.solPrim[self.varIdxs, :] = self.decodeSol(self.code)


	# def calcProjector(self, romDomain):
	# 	"""
	# 	Compute RHS projection operator
	# 	"""

	# 	if romDomain.hyperReduc:
	# 		raise ValueError("calcProjector() for gappy POD not implemented yet")

	# 	else:
			
	def calcDSol(self, resJacob, res):

		# calculate test basis
		# TODO: this is not valid for scalar POD, another reason to switch to C ordering of resJacob
		self.testBasis = (resJacob.toarray() / self.normFacProfCons.ravel(order="F")[:,None]) @ self.trialBasisFScaled

		

		# compute W^T * W
		LHS = self.testBasis.T @ self.testBasis
		RHS = -self.testBasis.T @ res.ravel(order="F")

		# linear solve
		dCode = np.linalg.solve(LHS, RHS)
		pdb.set_trace()
		
		return dCode, LHS, RHS


	def updateSol(self, solDomain):
		"""
		Update primitive solution after code has been updated
		"""

		solDomain.solInt.solPrim[self.varIdxs,:] = self.decodeSol(self.code)
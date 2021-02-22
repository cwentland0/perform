from perform.rom.linearProjROM.linearProjROM import linearProjROM
from perform.rom.ModelAdaption.adapt import adapt
import numpy as np
from scipy.sparse import csr_matrix

# TODO: could move some of these functions to linearProjROM and just branch if targeting cons vars or prim vars

class linearGalerkinProj(linearProjROM):
	"""
	Class for linear decoder and Galerkin projection
	Trial basis is assumed to represent the conserved variables (see SPLSVT for primitive variable representation)
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		super().__init__(modelIdx, romDomain, solver, solDomain)

		self.testBasis = self.trialBasis

		self.calcProjector(romDomain, runCalc=True)

		if romDomain.adaptiveROM: self.adapt = adapt(self, solver, romDomain)

	def decodeSol(self, code):
		"""
		Compute full decoding of conservative solution, including decentering and denormalization
		"""

		solCons = self.applyTrialBasis(code)
		solCons = self.standardizeData(solCons, 
									   normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons,
									   center=True, centProf=self.centProfCons, inverse=True)
		return solCons


	def initFromCode(self, code0, solDomain, solver):
		"""
		Initialize full-order conservative solution from input low-dimensional state
		"""

		self.code = code0.copy()
		solDomain.solInt.solCons[self.varIdxs, :] = self.decodeSol(self.code)


	def initFromSol(self, solDomain, solver):
		"""
		Initialize full-order conservative solution from projection of loaded full-order initial conditions
		"""

		solCons = self.standardizeData(solDomain.solInt.solCons[self.varIdxs, :], normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons, 
									   center=True, centProf=self.centProfCons, inverse=False)
		self.code = self.projectToLowDim(self.trialBasis, solCons, transpose=True)
		solDomain.solInt.solCons[self.varIdxs, :] = self.decodeSol(self.code)


	def calcProjector(self, romDomain, runCalc=False):
		"""
		Compute RHS projection operator
		NOTE: runCalc is kind of a stupid way to handle static vs. adaptive bases.
			  This method should generally be called with romDomain.adaptiveROM, but also needs to be calculated at init
		"""

		if runCalc:
			if romDomain.hyperReduc:
				# V^T * U * [S^T * U]^+
				self.projector = self.trialBasis.T @ self.hyperReducBasis @ np.linalg.pinv(self.hyperReducBasis[self.directHyperReducSampIdxs,:])

			else:
				# V^T
				self.projector = self.trialBasis.T
		else:
			pass


	def calcDCode(self, resJacob, res):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		trialBasisFordered = (self.trialBasis.reshape((self.numVars, -1, self.latentDim), order = 'C')).reshape(-1, self.latentDim, order = 'F')
		scaledTrialBasis   = trialBasisFordered*self.normFacProfCons.ravel(order="F")[:, np.newaxis]

		#V^{T}*H^{-1}*resJacob*H*V
		LHS = trialBasisFordered.T @ (resJacob / self.normFacProfCons.ravel(order="F")[:, np.newaxis]) @ scaledTrialBasis
		#V^{T}*H^{-1}*residual
		RHS = trialBasisFordered.T @ (res.ravel(order='F')[:, np.newaxis] / self.normFacProfCons.ravel(order="F")[:, np.newaxis])

		# #solving linear-system
		dCode = np.linalg.solve(LHS, RHS)
		return np.squeeze(dCode)


	def updateSol(self, solDomain):
		"""
		Update conservative solution after code has been updated
		"""
		# print(np.linalg.norm(solDomain.solInt.solCons, axis = 1))
		solDomain.solInt.solCons[self.varIdxs,:] = self.decodeSol(self.code)

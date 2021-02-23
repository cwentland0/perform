from perform.rom.projectionROM.linearProjROM.linearProjROM import linearProjROM

import numpy as np

class linearGalerkinProj(linearProjROM):
	"""
	Class for linear decoder and Galerkin projection
	Trial basis is assumed to represent the conserved variables
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if ((romDomain.timeIntegrator.timeType == "implicit") and (romDomain.timeIntegrator.dualTime)):
			raise ValueError("Galerkin is intended for conservative variable evolution, please set dualTime = False")

		super().__init__(modelIdx, romDomain, solver, solDomain)


	def calcProjector(self, solDomain):
		"""
		Compute RHS projection operator
		"""

		if self.hyperReduc:
			# V^T * U * [S^T * U]^+
			self.projector = self.trialBasis.T @ self.hyperReducBasis @ np.linalg.pinv(self.hyperReducBasis[self.directHyperReducSampIdxs,:])

		else:
			# V^T
			self.projector = self.trialBasis.T


	def calcDCode(self, resJacob, res):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		# TODO: should be calculated once
		scaledTrialBasis = self.trialBasis * self.normFacProfCons.ravel(order="C")[:,None]

		# TODO: using resJacob.toarray(), otherwise this operation returns type np.matrix, which is undesirable
		# 	need to figure out a more efficient method, if possible
		LHS = self.trialBasis.T @ (resJacob.toarray() / self.normFacProfCons.ravel(order="C")[:,None]) @ scaledTrialBasis
		RHS = self.trialBasis.T @ (res / self.normFacProfCons).ravel(order="C")

		dCode = np.linalg.solve(LHS, RHS)
		
		return dCode, LHS, RHS


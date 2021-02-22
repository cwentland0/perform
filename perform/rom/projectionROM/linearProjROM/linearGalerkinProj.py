from perform.rom.projectionROM.linearProjROM.linearProjROM import linearProjROM

import numpy as np
import pdb


class linearGalerkinProj(linearProjROM):
	"""
	Class for linear decoder and Galerkin projection
	Trial basis is assumed to represent the conserved variables (see SPLSVT for primitive variable representation)
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		super().__init__(modelIdx, romDomain, solver, solDomain)

		self.testBasis = self.trialBasis

		self.calcProjector(solDomain, runCalc=True)


	def calcProjector(self, solDomain, runCalc=False):
		"""
		Compute RHS projection operator
		NOTE: runCalc is kind of a stupid way to handle static vs. adaptive bases.
			  This method should generally be called with romDomain.adaptiveROM, but also needs to be calculated at init
		"""

		if runCalc:
			if self.hyperReduc:
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

		LHS = self.trialBasis.T @ (resJacob / self.normFacProfCons.ravel(order="C")[:,None]) @ (self.trialBasis * self.normFacProfCons.ravel(order="C")[:,None])
		RHS = self.trialBasis.T @ (res / self.normFacProfCons).ravel(order="C")

		dCode = np.linalg.solve(LHS, RHS)
		
		return dCode, LHS, RHS


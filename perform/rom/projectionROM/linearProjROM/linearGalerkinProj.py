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

		# TODO: any way to put this in projectionROM?

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


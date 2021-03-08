from perform.rom.projectionROM.linearProjROM.linearProjROM import linearProjROM

import numpy as np

class linearSPLSVTProj(linearProjROM):
	"""
	Class for linear decoder and SP-LSVT formulation
	Trial basis is assumed to represent the conserved variables
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		if (romDomain.timeIntegrator.timeType == "explicit"):
			raise ValueError("Explicit SP-LSVT not implemented yet")

		if ((romDomain.timeIntegrator.timeType == "implicit") and (not romDomain.timeIntegrator.dualTime)):
			raise ValueError("SP-LSVT is intended for primitive variable evolution, please use Galerkin or LSPG, or set dualTime = True")

		super().__init__(modelIdx, romDomain, solver, solDomain)


	def calcDCode(self, resJacob, res, solDomain):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		# TODO: add hyper-reduction

		# TODO: scaledTrialBasis should be calculated once
		scaledTrialBasis = self.trialBasis * self.normFacProfPrim.ravel(order="C")[:,None]

		# compute test basis
		# TODO: using resJacob.toarray(), otherwise this operation returns type np.matrix, which is undesirable
		# 	need to figure out a more efficient method, if possible
		testBasis = (resJacob.toarray() / self.normFacProfCons.ravel(order="C")[:,None]) @ scaledTrialBasis

		# LHS and RHS of Newton iteration
		LHS = testBasis.T @ testBasis
		RHS = testBasis.T @ (res / self.normFacProfCons).ravel(order="C")

		# linear solve
		dCode = np.linalg.solve(LHS, RHS)
		
		return dCode, LHS, RHS
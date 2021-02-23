from perform.rom.projectionROM.linearProjROM.linearProjROM import linearProjROM

import numpy as np

class linearLSPGProj(linearProjROM):
	"""
	Class for linear decoder and least-squares Petrov-Galerkin projection
	Trial basis is assumed to represent the conserved variables
	"""

	def __init__(self, modelIdx, romDomain, solver, solDomain):

		# I'm not going to code LSPG with explicit time integrator, it's a pointless exercise
		if (romDomain.timeIntegrator.timeType == "explicit"):
			raise ValueError("LSPG with an explicit time integrator deteriorates to Galerkin, please use Galerkin.")

		if romDomain.timeIntegrator.dualTime:
			raise ValueError("LSPG is intended for conservative variable evolution, please set dualTime = False")

		super().__init__(modelIdx, romDomain, solver, solDomain)


	def calcDCode(self, resJacob, res):
		"""
		Compute change in low-dimensional state for implicit scheme Newton iteration
		"""

		# TODO: add hyper-reduction

		# TODO: scaledTrialBasis should be calculated once
		scaledTrialBasis = self.trialBasis * self.normFacProfCons.ravel(order="C")[:,None]

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
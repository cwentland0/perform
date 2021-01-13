from pygems1d.rom.linearProjROM.linearProjROM import linearProjROM

import numpy as np
import pdb

# TODO: could move some of these functions to linearProjROM and just branch if targeting cons vars or prim vars

class linearGalerkinProj(linearProjROM):
	"""
	Class for linear decoder and Galerkin projection
	Trial basis is assumed to represent the conserved variables (see SPLSVT for primitive variable representation)
	"""

	def __init__(self, modelIdx, romDomain, solDomain, romDict, solver):

		super().__init__(modelIdx, romDomain, solDomain, romDict, solver)

		self.testBasis = self.trialBasis
		self.projector = self.trialBasis.T


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


	def calcProjector(self):
		"""
		Compute RHS projection operator
		"""

		if self.gappyPOD:
			raise ValueError("calcProjector() for gappy POD not implemented yet")

		else:
			pass


	def updateSol(self, solDomain):
		"""
		Update conservative solution after code has been updated
		"""

		solDomain.solInt.solCons[self.varIdxs,:] = self.decodeSol(self.code)

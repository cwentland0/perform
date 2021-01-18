from pygems1d.constants import realType
from pygems1d.solution.solutionPhys import solutionPhys 
from pygems1d.inputFuncs import parseBC

import numpy as np
from math import sin, pi
import pdb

class solutionBoundary(solutionPhys):
	"""
	Ghost cell solution
	"""

	def __init__(self, solDomain, solver, boundType):

		paramDict = solver.paramDict
		gas = solDomain.gasModel

		# this generally stores fixed/stagnation properties
		self.press, self.vel, self.temp, self.massFrac, self.rho, \
			self.pertType, self.pertPerc, self.pertFreq = parseBC(boundType, paramDict)

		assert (len(self.massFrac) == gas.numSpeciesFull), "Must provide mass fraction state for all species at boundary"
		assert (np.sum(self.massFrac) == 1.0), "Boundary mass fractions must sum to 1.0"

		self.CpMix 		= gas.calcMixCp(self.massFrac[gas.massFracSlice])
		self.RMix 		= gas.calcMixGasConstant(self.massFrac[gas.massFracSlice])
		self.gamma 		= gas.calcMixGamma(self.RMix, self.CpMix)
		self.enthRefMix = gas.calcMixEnthRef(self.massFrac[gas.massFracSlice])

		# this will be updated at each iteration, just initializing now
		# TODO: number of ghost cells should not always be one
		solDummy = np.ones((gas.numEqs,1), dtype=realType)
		super().__init__(solDomain, solDummy, 1, solver)
		self.solPrim[3:,0] = self.massFrac[gas.massFracSlice]

	
	def calcPert(self, t):
		"""
		Compute sinusoidal perturbation factor 
		"""

		# TODO: add phase offset

		pert = 0.0
		for f in self.pertFreq:
			pert += sin(2.0 * pi * self.pertFreq * t)
		pert *= self.pertPerc 

		return pert

	def calcBoundaryState(self, solver, solPrim=None, solCons=None):
		"""
		Run boundary calculation and update ghost cell state
		Assumed that boundary function sets primitive state
		"""

		self.boundFunc(solver, solPrim, solCons)
		self.updateState(fromCons = False)
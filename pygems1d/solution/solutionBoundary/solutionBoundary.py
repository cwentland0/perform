from pygems1d.constants import realType
from pygems1d.solution.solutionPhys import solutionPhys 
from pygems1d.inputFuncs import parseBC

import numpy as np
from math import sin, pi

class solutionBoundary(solutionPhys):
	"""
	Ghost cell solution
	"""

	def __init__(self, solver, boundType):

		paramDict = solver.paramDict
		gas = solver.gasModel

		# this generally stores fixed/stagnation properties
		self.press, self.vel, self.temp, self.massFrac, self.rho, \
			self.pertType, self.pertPerc, self.pertFreq = parseBC(boundType, paramDict)

		assert (len(self.massFrac) == gas.numSpeciesFull), "Must provide mass fraction state for all species at boundary"
		self.CpMix 		= gas.calcMixCp(self.massFrac[:-1])
		self.RMix 		= gas.calcMixGasConstant(self.massFrac[:-1])
		self.gamma 		= gas.calcMixGamma(self.RMix, self.CpMix)
		self.enthRefMix = gas.calcMixEnthRef(self.massFrac[:-1])

		# this will be updated at each iteration, just initializing now
		solDummy = np.ones((gas.numEqs,1), dtype=realType)
		super().__init__(solDummy, solDummy, 1, solver)
		self.solPrim[3:,0] = self.massFrac[:-1]

	
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
		self.updateState(solver.gasModel, fromCons = False)
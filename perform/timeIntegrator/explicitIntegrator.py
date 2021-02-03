import perform.constants as const
from perform.timeIntegrator.timeIntegrator import timeIntegrator

import numpy as np
import pdb

class explicitIntegrator(timeIntegrator):
	"""
	Base class for explicit time integrators
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType = "explicit"

		self.dualTime 	= False
		self.adaptDTau 	= False

class rkExplicit(explicitIntegrator):
	"""
	Low-memory explicit Runge-Kutta scheme
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		
		self.subiterMax = self.timeOrder
		self.coeffs = 1.0 / np.arange(1, self.timeOrder+1, dtype=const.realType)[::-1]


	def solveSolChange(self, rhs):
		dSol = self.dt * self.coeffs[self.subiter-1] * rhs
		return dSol


# TODO: Add "integrator" for ROM models that just update from one state to the next,
# 		presumably calls some generic update method.
#		This way, can still have a code history for methods that need the history but don't
# 		need numerical time integration (e.g. TCN)
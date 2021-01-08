import pygems1d.constants as const
from pygems1d.timeIntegrator.timeIntegrator import timeIntegrator

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
		self.coeffs = [1.0 / np.arange(1, self.timeOrder+1, dtype=const.realType)[::-1]]


	def solveSolChange(self, rhs):
		dSol = self.dt * self.coeffs[0][self.subiter-1] * rhs
		return dSol
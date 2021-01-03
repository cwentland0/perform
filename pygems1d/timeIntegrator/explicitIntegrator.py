import pygems1d.constants as const
from pygems1d.timeIntegrator.timeIntegrator import timeIntegrator
from pygems1d.spaceSchemes import calcRHS

import numpy as np

class explicitIntegrator(timeIntegrator):
	"""
	Base class for explicit time integrators
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType = "explicit"

		self.dualTime 	= False
		self.adaptDTau 	= False

	def advanceSubiter(self, solDomain, solROM, solver):

		calcRHS(solDomain, solver)

		# compute change in solution/code, advance solution/code
		if (solver.calcROM):
			solROM.mapRHSToModels(solDomain.solInt)
			solROM.calcRHSProjection()
			solROM.advanceSubiter(solDomain.solInt, solver)
		else:
			dSol = self.solveSolChange(solDomain.solInt.RHS)
			solDomain.solInt.solCons = solDomain.solInt.solHistCons[0] + dSol
				
		solDomain.solInt.updateState(solver.gasModel)

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
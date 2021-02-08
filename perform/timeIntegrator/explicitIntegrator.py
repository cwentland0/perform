import perform.constants as const
from perform.timeIntegrator.timeIntegrator import timeIntegrator

import numpy as np
import time
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

########## RUNGE-KUTTA INTEGRATORS ##########

class rkExplicit(explicitIntegrator):
	"""
	Explicit Runge-Kutta schemes
	"""

	def __init__(self, paramDict):

		super().__init__(paramDict)

		self.rkRHS = [None] * self.subiterMax # subiteration RHS history


	def solveSolChange(self, rhs):
		"""
		Either compute intermediate step or final physical time step
		"""

		self.rkRHS[self.subiter] = rhs.copy()

		if (self.subiter == (self.subiterMax-1)):
			dSol = self.solveSolChangeIter(rhs)
		else:
			dSol = self.solveSolChangeSubiter(rhs)

		dSol *= self.dt

		return dSol


	def solveSolChangeSubiter(self, rhs):
		"""
		Change in intermediate solution for subiteration
		"""

		dSol = np.zeros(rhs.shape, dtype=const.realType)
		for rkIter in range(self.subiter+1):
			rkA = self.rkAVals[self.subiter+1][rkIter]
			if (rkA != 0.0):
				dSol += rkA * self.rkRHS[rkIter]

		return dSol


	def solveSolChangeIter(self, rhs):
		"""
		Change in physical solution
		"""
		
		dSol = np.zeros(rhs.shape, dtype=const.realType)
		for rkIter in range(self.subiterMax):
			rkB = self.rkBVals[rkIter]
			if (rkB != 0.0):
				dSol += rkB * self.rkRHS[rkIter]

		return dSol


class classicRK4(rkExplicit):
	"""
	Classic explicit RK4 scheme
	"""

	def __init__(self, paramDict):

		self.subiterMax = 4

		super().__init__(paramDict)

		# warn user
		if (self.timeOrder != 4):
			print("classicRK4 is fourth-order accurate, but you set timeOrder = "+str(self.timeOrder))
			print("Continuing, set timeOrder = 4 to get rid of this warning")
			time.sleep(0.5)

		self.rkAVals = [[0.0, 0.0, 0.0, 0.0],
						[0.5, 0.0, 0.0, 0.0],
				   		[0.0, 0.5, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0]]
		self.rkBVals = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
		self.rkCVals = [0.0, 0.5, 0.5, 1.0]


class sspRK3(rkExplicit):
	"""
	Strong stability-preserving explicit RK3 scheme
	"""

	def __init__(self, paramDict):

		self.subiterMax = 3

		super().__init__(paramDict)

		if (self.timeOrder != 3):
			print("sspRK3 is third-order accurate, but you set timeOrder = "+str(self.timeOrder))
			print("Continuing, set timeOrder = 3 to get rid of this warning")
			time.sleep(0.5)

		self.rkAVals = [[0.0,  0.0,  0.0],
						[1.0,  0.0,  0.0],
				   		[0.25, 0.25, 0.0]]
		self.rkBVals = [1.0/6.0, 1.0/6.0, 2.0/3.0]
		self.rkCVals = [0.0, 1.0, 0.5]


class jamesonLowStore(rkExplicit):
	"""
	"Low-storage" class of RK schemes by Jameson
	Not actually appropriate for unsteady problems, supposedly
	Not actually low-storage to work with general RK format, 
		just maintained here for consistency with old code
	"""

	def __init__(self, paramDict):

		timeOrder 	= int(paramDict["timeOrder"]) # have to get this early to set subiterMax
		self.subiterMax = timeOrder

		super().__init__(paramDict)

		# if you add another row and column to rkAVals, update this assertion
		assert (timeOrder <= 4), "Jameson low-storage RK scheme for order "+str(timeOrder)+" not yet implemented"
		
		self.rkAVals = np.array([[0.0,  0.0,   0.0, 0.0],
								 [0.25, 0.0,   0.0, 0.0],
				   				 [0.0,  1./3., 0.0, 0.0],
								 [0.0,  0.0,   0.5, 0.0]])

		self.rkAVals = self.rkAVals[-timeOrder:,-timeOrder:]

		self.rkBVals = np.zeros(timeOrder, dtype=const.realType)
		self.rkBVals[-1] = 1.0
		self.rkCVals = np.zeros(timeOrder, dtype=const.realType)

########## END RUNGE-KUTTA INTEGRATORS ##########


# TODO: Add "integrator" for ROM models that just update from one state to the next,
# 		presumably calls some generic update method.
#		This way, can still have a code history for methods that need the history but don't
# 		need numerical time integration (e.g. TCN)
import numpy as np
import constants
from inputFuncs import catchInput
from solution import solutionPhys, boundaries
from romClasses import solutionROM
from spaceSchemes import calcRHS
import pdb

# TODO: could have another level of hierarchy for explicit, implicit, and implicit+dual integrators


class timeIntegrator:
	"""
	Base class for time integrators
	"""

	def __init__(self, paramDict):

		self.dt 		= float(paramDict["dt"])		# physical time step
		self.numSteps 	= int(paramDict["numSteps"])	# total number of physical time iterations
		self.timeScheme = str(paramDict["timeScheme"])	# time integration scheme
		self.timeOrder 	= int(paramDict["timeOrder"])	# time integration order of accuracy
		assert (self.timeOrder >= 1), "timeOrder only accepts positive integer values."

		self.runSteady 	= catchInput(paramDict, "runSteady", False) # run "steady" simulation

		self.iter 		= 1 	# iteration number for current run
		self.subiter 	= 1		# subiteration number for multi-stage schemes
		self.timeIter 	= 1 	# physical time iteration number for restarted solutions
		self.subiterMax = None	# maximum number of subiterations for multi-stage schemes

	def updateSubiter(self):
		if (self.subiter == self.subiterMax):
			self.subiter = 1
		else:
			self.subiter += 1

	def updateIter(self):
		self.iter += 1
		self.timeIter += 1

	def advanceIter(self, solPhys: solutionPhys, solROM: solutionROM, bounds: boundaries, solver):

		for subiter in range(self.subiterMax):	
			self.advanceSubiter(solPhys, solROM, bounds, solver)

		solPhys.updateSolHist() 
		self.updateIter()

class explicitIntegrator(timeIntegrator):

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType = "explicit"

	def advanceSubiter(self, solPhys: solutionPhys, solROM: solutionROM, bounds: boundaries, solver):

		# compute RHS function
		calcRHS(solPhys, bounds, solver)

		# compute change in solution/code, advance solution/code
		if (solver.calcROM):
			solROM.mapRHSToModels(solPhys)
			solROM.calcRHSProjection()
			solROM.advanceSubiter(solPhys, solver)
		else:
			dSol = self.solveSolChange(solPhys.RHS)
			solPhys.solCons = solPhys.solHistCons[0] + dSol
		
		# pdb.set_trace()
		
		solPhys.updateState(solver.gasModel)
		self.updateSubiter()

class RKExplicit(explicitIntegrator):
	"""
	Low-memory explicit Runge-Kutta scheme
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		
		self.subiterMax = self.timeOrder
		self.coeffs = 1.0 / np.arange(1, self.timeOrder+1, dtype=constants.realType)[::-1]


	def solveSolChange(self, rhs):
		dSol = self.dt * self.coeffs[self.subiter-1] * rhs
		return dSol

# class LMSImplicitDual(timeIntegrator):
# 	"""
# 	Implicit linear multi-step scheme (backwards difference formula) with dual time stepping
# 	"""

# 	def __init__(self, paramDict):
# 		super().__init__(paramDict)

# 		self.timeType 		= "implicit"
# 		self.subiterMax		= catchInput(paramDict, "subiterMax", constants.subiterMaxImpDefault)
# 		self.resTol 		= catchInput(paramDict, "resTol", constants.l2ResTolDefault)
# 		self.dtau 			= catchInput(paramDict, "dtau", constants.dtauDefault)

# 	def calcResidual(self, solHist, rhs):
		
# 		# cold start 
# 		if (self.iter < self.timeOrder):
# 			timeOrder = self.iter
# 		else:
# 			timeOrder = self.timeOrder
		
# 		if (timeOrder == 1):
# 			residual = solHist[0] - solHist[1]
# 		elif (timeOrder == 2):
# 			residual = 1.5*solHist[0] - 2.*solHist[1] + 0.5*solHist[2]
# 		elif (timeOrder == 3):
# 			residual = 11./6.*solHist[0] - 3.*solHist[1] + 1.5*solHist[2] -1./3.*solHist[3]
# 		elif (timeOrder == 4):
# 			residual = 25./12.*solHist[0] - 4.*solHist[1] + 3.*solHist[2] -4./3.*solHist[3] + 0.25*solHist[4]
# 		else:
# 			raise ValueError(str(self.timeOrder)+"th-order accurate scheme not implemented for "+self.timeScheme+" scheme")
		
# 		residual = -(residual / self.dt) + rhs

# 		return residual

# class sspRKExplicit(timeIntegrator):

# 	def __init__(self, params: parameters):

# 		super().__init__(params)


import constants
from inputFuncs import catchInput
from solution import solutionPhys, boundaries
from romClasses import solutionROM
from spaceSchemes import calcRHS
from Jacobians import calcDResDSolPrim
import numpy as np
from scipy.sparse.linalg import spsolve
import pdb

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

	def advanceIter(self, solPhys: solutionPhys, solROM: solutionROM, bounds: boundaries, solver):

		if (not solver.timeIntegrator.runSteady): print("Iteration "+str(self.iter))

		for self.subiter in range(1, self.subiterMax+1):	
			self.advanceSubiter(solPhys, solROM, bounds, solver)

			if (self.timeType == "implicit"):
				solPhys.resOutput(solver)
				if (solPhys.resNormL2 < self.resTol): break

		solPhys.updateSolHist() 

class explicitIntegrator(timeIntegrator):
	"""
	Base class for explicit time integrators
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType = "explicit"

		self.dualTime 	= False
		self.adaptDTau 	= False

	def advanceSubiter(self, solPhys: solutionPhys, solROM: solutionROM, bounds: boundaries, solver):

		calcRHS(solPhys, bounds, solver)

		# compute change in solution/code, advance solution/code
		if (solver.calcROM):
			solROM.mapRHSToModels(solPhys)
			solROM.calcRHSProjection()
			solROM.advanceSubiter(solPhys, solver)
		else:
			dSol = self.solveSolChange(solPhys.RHS)
			solPhys.solCons = solPhys.solHistCons[0] + dSol
				
		solPhys.updateState(solver.gasModel)

class implicitIntegrator(timeIntegrator):
	"""
	Base class for implicit time integrators
	Solves implicit system via Newton's method
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType 		= "implicit"
		self.subiterMax		= catchInput(paramDict, "subiterMax", constants.subiterMaxImpDefault)
		self.resTol 		= catchInput(paramDict, "resTol", constants.l2ResTolDefault)

		# dual time-stepping, robustness controls
		self.dualTime 		= catchInput(paramDict, "dualTime", True)
		self.dtau 			= catchInput(paramDict, "dtau", constants.dtauDefault)
		if (self.dualTime):
			self.adaptDTau 	= catchInput(paramDict, "adaptDTau", False)
		else:
			self.adaptDTau 	= False
		self.CFL 			= catchInput(paramDict, "CFL", constants.CFLDefault) 	# reference CFL for advective control of dtau
		self.VNN 			= catchInput(paramDict, "VNN", constants.VNNDefault) 	# von Neumann number for diffusion control of dtau
		self.refConst 		= catchInput(paramDict, "refConst", [None])  			# constants for limiting dtau	
		self.relaxConst 	= catchInput(paramDict, "relaxConst", [None]) 			#

	def advanceSubiter(self, solPhys: solutionPhys, solROM: solutionROM, bounds: boundaries, solver):

		calcRHS(solPhys, bounds, solver) 

		# compute discretized system residual
		solPhys.res = self.calcResidual(solPhys.solHistCons, solPhys.RHS)

		# compute Jacobian of residual
		# TODO: new Jacobians, state update for non-dual time 
		resJacob = calcDResDSolPrim(solPhys, bounds, solver)

		# solve linear system 
		dSol = spsolve(resJacob, solPhys.res.flatten('C'))

		# update state
		solPhys.solPrim += dSol.reshape((solver.mesh.numCells, solver.gasModel.numEqs), order='C')
		solPhys.updateState(solver.gasModel, fromCons = False)
		solPhys.solHistCons[0] = solPhys.solCons.copy() 
		solPhys.solHistPrim[0] = solPhys.solPrim.copy() 

		# borrow solPhys.res to store linear solve residual	
		res = resJacob @ dSol - solPhys.res.flatten('C')
		solPhys.res = np.reshape(res, (solver.mesh.numCells, solver.gasModel.numEqs), order='C')

class rkExplicit(explicitIntegrator):
	"""
	Low-memory explicit Runge-Kutta scheme
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		
		self.subiterMax = self.timeOrder
		self.coeffs = [1.0 / np.arange(1, self.timeOrder+1, dtype=constants.realType)[::-1]]


	def solveSolChange(self, rhs):
		dSol = self.dt * self.coeffs[0][self.subiter-1] * rhs
		return dSol

class bdf(implicitIntegrator):
	"""
	Backwards difference formula (up to fourth-order)
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)

		self.coeffs = [None]*4
		self.coeffs[0] = np.array([1.0, -1.0], dtype=constants.realType)
		self.coeffs[1] = np.array([1.5, -2.0, 0.5], dtype=constants.realType)
		self.coeffs[2] = np.array([11./16., -3.0, 1.5, -1./3.], dtype=constants.realType)
		self.coeffs[3] = np.array([25./12., -4.0, 3.0, -4./3., 0.25], dtype=constants.realType)
		assert (self.timeOrder <= 4), str(self.timeOrder)+"th-order accurate scheme not implemented for "+self.timeScheme+" scheme"

	def calcResidual(self, solHist, rhs):
		
		timeOrder = min(self.iter, self.timeOrder) 	# cold start
		coeffs = self.coeffs[timeOrder-1]

		residual = coeffs[0] * solHist[0]
		for iterIdx in range(1, timeOrder+1):
			residual += coeffs[iterIdx] * solHist[iterIdx]
		
		residual = -(residual / self.dt) + rhs

		return residual


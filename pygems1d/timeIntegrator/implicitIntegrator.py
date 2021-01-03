import pygems1d.constants as const
from pygems1d.inputFuncs import catchInput
from pygems1d.timeIntegrator.timeIntegrator import timeIntegrator
from pygems1d.spaceSchemes import calcRHS
from pygems1d.Jacobians import calcDResDSolPrim

import numpy as np
from scipy.sparse.linalg import spsolve

class implicitIntegrator(timeIntegrator):
	"""
	Base class for implicit time integrators
	Solves implicit system via Newton's method
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)
		self.timeType 		= "implicit"
		self.subiterMax		= catchInput(paramDict, "subiterMax", const.subiterMaxImpDefault)
		self.resTol 		= catchInput(paramDict, "resTol", const.l2ResTolDefault)

		# dual time-stepping, robustness controls
		self.dualTime 		= catchInput(paramDict, "dualTime", True)
		self.dtau 			= catchInput(paramDict, "dtau", const.dtauDefault)
		if (self.dualTime):
			self.adaptDTau 	= catchInput(paramDict, "adaptDTau", False)
		else:
			self.adaptDTau 	= False
		self.CFL 			= catchInput(paramDict, "CFL", const.CFLDefault) 	# reference CFL for advective control of dtau
		self.VNN 			= catchInput(paramDict, "VNN", const.VNNDefault) 	# von Neumann number for diffusion control of dtau
		self.refConst 		= catchInput(paramDict, "refConst", [None])  			# constants for limiting dtau	
		self.relaxConst 	= catchInput(paramDict, "relaxConst", [None]) 			#

	def advanceSubiter(self, solDomain, solROM, solver):

		solInt = solDomain.solInt

		calcRHS(solDomain, solver) 

		# compute discretized system residual
		solInt.res = self.calcResidual(solInt.solHistCons, solInt.RHS)

		# compute Jacobian of residual
		# TODO: new Jacobians, state update for non-dual time 
		resJacob = calcDResDSolPrim(solDomain, solver)

		# solve linear system 
		dSol = spsolve(resJacob, solInt.res.flatten('C'))

		# update state
		solInt.solPrim += dSol.reshape((solver.mesh.numCells, solver.gasModel.numEqs), order='C')
		solInt.updateState(solver.gasModel, fromCons = False)
		solInt.solHistCons[0] = solInt.solCons.copy() 
		solInt.solHistPrim[0] = solInt.solPrim.copy() 

		# borrow solInt.res to store linear solve residual	
		res = resJacob @ dSol - solInt.res.flatten('C')
		solInt.res = np.reshape(res, (solver.mesh.numCells, solver.gasModel.numEqs), order='C')

class bdf(implicitIntegrator):
	"""
	Backwards difference formula (up to fourth-order)
	"""

	def __init__(self, paramDict):
		super().__init__(paramDict)

		self.coeffs = [None]*4
		self.coeffs[0] = np.array([1.0, -1.0], dtype=const.realType)
		self.coeffs[1] = np.array([1.5, -2.0, 0.5], dtype=const.realType)
		self.coeffs[2] = np.array([11./16., -3.0, 1.5, -1./3.], dtype=const.realType)
		self.coeffs[3] = np.array([25./12., -4.0, 3.0, -4./3., 0.25], dtype=const.realType)
		assert (self.timeOrder <= 4), str(self.timeOrder)+"th-order accurate scheme not implemented for "+self.timeScheme+" scheme"

	def calcResidual(self, solHist, rhs):
		
		timeOrder = min(self.iter, self.timeOrder) 	# cold start
		coeffs = self.coeffs[timeOrder-1]

		residual = coeffs[0] * solHist[0]
		for iterIdx in range(1, timeOrder+1):
			residual += coeffs[iterIdx] * solHist[iterIdx]
		
		residual = -(residual / self.dt) + rhs

		return residual
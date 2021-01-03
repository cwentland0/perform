from pygems1d.constants import realType, resNormPrimDefault
from pygems1d.solution.solutionPhys import solutionPhys
import numpy as np

class solutionInterior(solutionPhys):
	"""
	Solution of interior domain
	"""

	def __init__(self, solPrimIn, solConsIn, solver):
		super().__init__(solPrimIn, solConsIn, solver.mesh.numCells, solver)

		gas = solver.gasModel
		timeInt = solver.timeIntegrator
		numCells = solver.mesh.numCells 

		self.source = np.zeros((numCells, gas.numSpecies), dtype=realType)	# reaction source term
		self.RHS 	= np.zeros((numCells, gas.numEqs), dtype=realType)		# RHS function

		# residual, time history, residual normalization for implicit methods
		if (timeInt.timeType == "implicit"):

			if ((timeInt.dualTime) and (timeInt.adaptDTau)):
				self.srf 	= np.zeros(numCells, dtype=realType)

			self.res 			= np.zeros((numCells, gas.numEqs), dtype=realType)
			self.solHistCons 	= [np.zeros((numCells, gas.numEqs), dtype=realType)]*(timeInt.timeOrder+1)
			self.solHistPrim 	= [np.zeros((numCells, gas.numEqs), dtype=realType)]*(timeInt.timeOrder+1)

			self.resNormL2 = 0.0
			self.resNormL1 = 0.0

			# if nothing was provided
			if ((len(solver.resNormPrim) == 1) and (solver.resNormPrim[0] == None)):
				solver.resNormPrim = [None]*gas.numEqs
			# check user input
			else:
				assert(len(solver.resNormPrim) == gas.numEqs)

			# replace any None's with defaults
			for varIdx in range(gas.numEqs):
				if (solver.resNormPrim[varIdx] == None):
					# 0: pressure, 1: velocity, 2: temperature, >=3: species
					solver.resNormPrim[varIdx] = resNormPrimDefault[min(varIdx,3)]

		# add bulk velocity if required
		if (solver.velAdd != 0.0):
			self.solPrim[:,1] += solver.velAdd
		
		self.updateState(gas, fromCons=False)

		# initializing time history
		self.solHistCons = [self.solCons.copy()]*(timeInt.timeOrder+1)
		self.solHistPrim = [self.solPrim.copy()]*(timeInt.timeOrder+1)

		# snapshot storage matrices, store initial condition
		if solver.primOut: 
			self.primSnap = np.zeros((numCells, gas.numEqs, solver.numSnaps+1), dtype=realType)
			self.primSnap[:,:,0] = solPrimIn.copy()
		if solver.consOut: 
			self.consSnap = np.zeros((numCells, gas.numEqs, solver.numSnaps+1), dtype=realType)
			self.consSnap[:,:,0] = solConsIn.copy()
		if solver.sourceOut: self.sourceSnap = np.zeros((numCells, gas.numSpecies, solver.numSnaps+1), dtype=realType)
		if solver.RHSOut:  self.RHSSnap  = np.zeros((numCells, gas.numEqs, solver.numSnaps), dtype=realType)
		
	
	def updateSolHist(self):
		"""
		Update time history of solution for implicit time integration
		"""
	
		self.solHistCons[1:] = self.solHistCons[:-1]
		self.solHistPrim[1:] = self.solHistPrim[:-1]
		self.solHistCons[0] = self.solCons
		self.solHistPrim[0] = self.solPrim


	def resOutput(self, solver):
		"""
		Calculate and print meaningful residual norms
		For unsteady implicit solve, this is the linear solve residual
		For "steady" solve, this is the change in solution between outer iterations
		"""

		# TODO: do some decent formatting on output

		if (solver.timeIntegrator.runSteady):
			if (solver.timeIntegrator.subiter == 1):
				res = self.solHistPrim[1] - self.solHistPrim[2]
				iterOut = solver.timeIntegrator.timeIter
			else:
				return
		else:
			res = self.res
			iterOut = solver.timeIntegrator.subiter
		
		resAbs = np.abs(res)

		# L2 norm
		resNormL2 = np.sum(np.square(resAbs), axis=0)
		resNormL2[:] /= res.shape[0]
		resNormL2 /= np.square(solver.resNormPrim)
		resNormL2 = np.sqrt(resNormL2)
		resNormL2 = np.mean(resNormL2)
		resOutL2 = np.log10(resNormL2)
			
		# L1 norm
		resNormL1 = np.sum(resAbs, axis=0)
		resNormL1[:] /= res.shape[0]
		resNormL1 /= solver.resNormPrim
		resNormL1 = np.mean(resNormL1)
		resOutL1 = np.log10(resNormL1)

		print(str(iterOut)+":\tL2 norm: "+str(resOutL2)+",\tL1 norm:"+str(resOutL1))

		self.resNormL2 = resNormL2
		self.resNormL1 = resNormL1
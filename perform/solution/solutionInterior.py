import perform.constants as const
from perform.constants import realType, resNormPrimDefault
from perform.solution.solutionPhys import solutionPhys

import numpy as np
import os
import pdb

class solutionInterior(solutionPhys):
	"""
	Solution of interior domain
	"""

	def __init__(self, gas, solPrimIn, solver, timeInt):
		super().__init__(gas, solPrimIn, solver.mesh.numCells)

		gas = self.gasModel
		numCells = solver.mesh.numCells 

		self.source = np.zeros((gas.numSpecies,numCells), dtype=realType)	# reaction source term
		self.RHS 	= np.zeros((gas.numEqs,numCells), dtype=realType)		# RHS function

		# add bulk velocity and update state if requested
		if (solver.velAdd != 0.0):
			self.solPrim[1,:] += solver.velAdd
			self.updateState(fromCons=False)

		# initializing time history
		self.solHistCons = [self.solCons.copy()] * (timeInt.timeOrder+1)
		self.solHistPrim = [self.solPrim.copy()] * (timeInt.timeOrder+1)

		# snapshot storage matrices, store initial condition
		if solver.primOut: 
			self.primSnap = np.zeros((gas.numEqs, numCells, solver.numSnaps+1), dtype=realType)
			self.primSnap[:,:,0] = self.solPrim.copy()
		if solver.consOut: 
			self.consSnap = np.zeros((gas.numEqs, numCells, solver.numSnaps+1), dtype=realType)
			self.consSnap[:,:,0] = self.solCons.copy()

		# these don't include the source/RHS associated with the final solution
		# TODO: calculate at final solution? Doesn't really seem worth the bother to me
		if solver.sourceOut: self.sourceSnap = np.zeros((gas.numSpecies, numCells, solver.numSnaps), dtype=realType)
		if solver.RHSOut:  self.RHSSnap  = np.zeros((gas.numEqs, numCells, solver.numSnaps), dtype=realType)

		if ((timeInt.timeType == "implicit") or (solver.runSteady)):
			# norm normalization constants
			# TODO: will need a normalization constant for conservative residual when it's implemented
			if ((len(solver.resNormPrim) == 1) and (solver.resNormPrim[0] == None)):
				solver.resNormPrim = [None]*gas.numEqs
			else:
				assert(len(solver.resNormPrim) == gas.numEqs)
			for varIdx in range(gas.numEqs):
				if (solver.resNormPrim[varIdx] == None):
					# 0: pressure, 1: velocity, 2: temperature, >=3: species
					solver.resNormPrim[varIdx] = resNormPrimDefault[min(varIdx,3)]

			# residual norm storage
			if (timeInt.timeType == "implicit"):

				self.res 			= np.zeros((gas.numEqs,numCells), dtype=realType)
				self.resNormL2 		= 0.0
				self.resNormL1 		= 0.0
				self.resNormHistory = np.zeros((solver.numSteps,2), dtype=realType) 

				if ((timeInt.dualTime) and (timeInt.adaptDTau)):
					self.srf 	= np.zeros(numCells, dtype=realType)

			# "steady" convergence measures
			if solver.runSteady:

				self.dSolNormL2 		= 0.0
				self.dSolNormL1 		= 0.0
				self.dSolNormHistory 	= np.zeros((solver.numSteps,2), dtype=realType)
		
	
	def updateSolHist(self):
		"""
		Update time history of solution for multi-stage time integrators
		"""
	
		self.solHistCons[1:] = self.solHistCons[:-1]
		self.solHistPrim[1:] = self.solHistPrim[:-1]
		self.solHistCons[0]  = self.solCons.copy()
		self.solHistPrim[0]  = self.solPrim.copy()


	def updateSnapshots(self, solver):

		storeIdx = int((solver.iter - 1) / solver.outInterval) + 1

		if solver.primOut: 		self.primSnap[:,:,storeIdx] 	= self.solPrim
		if solver.consOut: 		self.consSnap[:,:,storeIdx] 	= self.solCons
		if solver.sourceOut:	self.sourceSnap[:,:,storeIdx-1] = self.source
		if solver.RHSOut:  		self.RHSSnap[:,:,storeIdx-1]  	= self.RHS


	def writeSnapshots(self, solver):
		"""
		Save snapshot matrices to disk
		"""

		finalIdx = int((solver.iter - 1) / solver.outInterval) + 1 # accounts for failed simulation dump

		if solver.primOut:
			solPrimFile = os.path.join(const.unsteadyOutputDir, "solPrim_"+solver.simType+".npy")
			np.save(solPrimFile, self.primSnap[:,:,:finalIdx])
		if solver.consOut:
			solConsFile = os.path.join(const.unsteadyOutputDir, "solCons_"+solver.simType+".npy")
			np.save(solConsFile, self.consSnap[:,:,:finalIdx]) 
		if solver.sourceOut:
			sourceFile = os.path.join(const.unsteadyOutputDir, "source_"+solver.simType+".npy")
			np.save(sourceFile, self.sourceSnap[:,:,:finalIdx-1])
		if solver.RHSOut:
			solRHSFile = os.path.join(const.unsteadyOutputDir, "solRHS_"+solver.simType+".npy")
			np.save(solRHSFile, self.RHSSnap[:,:,:finalIdx-1]) 


	def writeRestartFile(self, solver):
		"""
		Write restart files containing primitive and conservative fields, and associated physical time
		"""

		# TODO: write previous time step(s) for multi-step methods, to preserve time accuracy at restart

		# write restart file to zipped file
		restartFile = os.path.join(const.restartOutputDir, "restartFile_"+str(solver.restartIter)+".npz")
		np.savez(restartFile, solTime = solver.solTime, solPrim = self.solPrim, solCons = self.solCons)

		# write iteration number files
		restartIterFile = os.path.join(const.restartOutputDir, "restartIter.dat")
		with open(restartIterFile, "w") as f:
			f.write(str(solver.restartIter)+"\n")

		restartPhysIterFile = os.path.join(const.restartOutputDir, "restartIter_"+str(solver.restartIter)+".dat")
		with open(restartPhysIterFile, "w") as f:
			f.write(str(solver.iter)+"\n")

		# iterate file count
		if (solver.restartIter < solver.numRestarts):
			solver.restartIter += 1
		else:
			solver.restartIter = 1


	def writeSteadyData(self, solver):

		# write norm data to ASCII file
		steadyFile = os.path.join(const.unsteadyOutputDir, "steadyConvergence.dat")
		if (solver.iter == 1):
			f = open(steadyFile,"w")
		else:
			f = open(steadyFile, "a")
		outString = ("%8i %18.14f %18.14f\n") % (solver.timeIter-1, self.dSolNormL2, self.dSolNormL1)
		f.write(outString)
		f.close()

		# write field data
		solPrimFile = os.path.join(const.unsteadyOutputDir, "solPrim_steady.npy")
		np.save(solPrimFile, self.solPrim)
		solConsFile = os.path.join(const.unsteadyOutputDir, "solCons_steady.npy")
		np.save(solConsFile, self.solCons)


	def calcDSolNorms(self, solver, timeType):
		"""
		Calculate and print solution change norms
		Note that output is ORDER OF MAGNITUDE of residual norm (i.e. 1e-X, where X is the order of magnitude)
		"""

		if (timeType == "implicit"):
			dSol = self.solHistPrim[0] - self.solHistPrim[1]
		else:
			# TODO: only valid for single-stage explicit schemes
			dSol = self.solPrim - self.solHistPrim[0]
			
		normL2, normL1 = self.calcNorms(dSol, solver.resNormPrim)

		normOutL2 = np.log10(normL2)
		normOutL1 = np.log10(normL1)
		outString = ("%8i:   L2: %18.14f,   L1: %18.14f") % (solver.timeIter, normOutL2, normOutL1)
		print(outString)

		self.dSolNormL2 = normL2
		self.dSolNormL1 = normL1
		self.dSolNormHistory[solver.iter-1, :] = [normL2, normL1]


	def calcResNorms(self, solver, subiter):
		"""
		Calculate and print linear solve residual norms		
		Note that output is ORDER OF MAGNITUDE of residual norm (i.e. 1e-X, where X is the order of magnitude)
		"""

		# TODO: pass conservative normalization factors if running conservative implicit solve
		normL2, normL1 = self.calcNorms(self.res, solver.resNormPrim)

		# don't print for "steady" solve
		if (not solver.runSteady):
			normOutL2 = np.log10(normL2)
			normOutL1 = np.log10(normL1)
			outString = (str(subiter)+":\tL2: %18.14f, \tL1: %18.14f") % (normOutL2, normOutL1)
			print(outString)

		self.resNormL2 = normL2
		self.resNormL1 = normL1
		self.resNormHistory[solver.iter-1, :] = [normL2, normL1]


	def calcNorms(self, arrIn, normFacs):
		"""
		Compute L1 and L2 norms of arrIn
		arrIn assumed to be in [numVars, numCells] order
		"""

		arrAbs = np.abs(arrIn)

		# L2 norm
		arrNormL2 = np.sum(np.square(arrAbs), axis=1)
		arrNormL2[:] /= arrIn.shape[1]
		arrNormL2 /= np.square(normFacs)
		arrNormL2 = np.sqrt(arrNormL2)
		arrNormL2 = np.mean(arrNormL2)
		
		# L1 norm
		arrNormL1 = np.sum(arrAbs, axis=1)
		arrNormL1[:] /= arrIn.shape[1]
		arrNormL1 /= normFacs
		arrNormL1 = np.mean(arrNormL1)
		
		return arrNormL2, arrNormL1
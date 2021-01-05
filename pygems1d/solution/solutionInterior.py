import pygems1d.constants as const
from pygems1d.constants import realType, resNormPrimDefault
from pygems1d.solution.solutionPhys import solutionPhys

import numpy as np
import os
import pdb

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

		# these don't include the source/RHS associated with the final solution
		# TODO: calculate at final solution? Doesn't really seem worth the bother to me
		if solver.sourceOut: self.sourceSnap = np.zeros((numCells, gas.numSpecies, solver.numSnaps), dtype=realType)
		if solver.RHSOut:  self.RHSSnap  = np.zeros((numCells, gas.numEqs, solver.numSnaps), dtype=realType)

		# residual storage, residual normalization for implicit methods
		if (timeInt.timeType == "implicit"):

			self.res 			= np.zeros((numCells, gas.numEqs), dtype=realType)
			self.resNormL2 		= 0.0
			self.resNormL1 		= 0.0
			self.resNormHistory = np.zeros((solver.timeIntegrator.numSteps,2), dtype=realType) 

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

			if ((timeInt.dualTime) and (timeInt.adaptDTau)):
				self.srf 	= np.zeros(numCells, dtype=realType)


		
	
	def updateSolHist(self):
		"""
		Update time history of solution for implicit time integration
		"""
	
		self.solHistCons[1:] = self.solHistCons[:-1]
		self.solHistPrim[1:] = self.solHistPrim[:-1]
		self.solHistCons[0] = self.solCons
		self.solHistPrim[0] = self.solPrim

	def updateSnapshots(self, solver):

		storeIdx = int((solver.timeIntegrator.iter - 1) / solver.outInterval) + 1

		if solver.primOut: 		self.primSnap[:,:,storeIdx] 	= self.solPrim
		if solver.consOut: 		self.consSnap[:,:,storeIdx] 	= self.solCons
		if solver.sourceOut:	self.sourceSnap[:,:,storeIdx-1] = self.source
		if solver.RHSOut:  		self.RHSSnap[:,:,storeIdx-1]  	= self.RHS

	def writeSnapshots(self, solver):
		"""
		Save snapshot matrices to disk
		"""

		if solver.primOut:
			solPrimFile = os.path.join(const.unsteadyOutputDir, "solPrim_"+solver.simType+".npy")
			np.save(solPrimFile, self.primSnap)
		if solver.consOut:
			solConsFile = os.path.join(const.unsteadyOutputDir, "solCons_"+solver.simType+".npy")
			np.save(solConsFile, self.consSnap) 
		if solver.sourceOut:
			sourceFile = os.path.join(const.unsteadyOutputDir, "source_"+solver.simType+".npy")
			np.save(sourceFile, self.sourceSnap)
		if solver.RHSOut:
			solRHSFile = os.path.join(const.unsteadyOutputDir, "solRHS_"+solver.simType+".npy")
			np.save(solRHSFile, self.RHSSnap) 

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
			f.write(str(solver.timeIntegrator.iter)+"\n")

		# iterate file count
		if (solver.restartIter < solver.numRestarts):
			solver.restartIter += 1
		else:
			solver.restartIter = 1

	def writeSteadyData(self, solver):

		# write norm data to ASCII file
		steadyFile = os.path.join(const.unsteadyOutputDir, "steadyConvergence.dat")
		if (solver.timeIntegrator.iter == 1):
			f = open(steadyFile,"w")
		else:
			f = open(steadyFile, "a")
		f.write(str(solver.timeIntegrator.iter)+"\t"+str(self.resNormL2)+"\t"+str(self.resNormL1)+"\n")
		f.close()

		# write field data
		solPrimFile = os.path.join(const.unsteadyOutputDir, "solPrim_steady.npy")
		np.save(solPrimFile, self.solPrim)
		solConsFile = os.path.join(const.unsteadyOutputDir, "solCons_steady.npy")
		np.save(solConsFile, self.solCons)

	def resOutput(self, solver):
		"""
		Calculate and print meaningful residual norms
		For unsteady implicit solve, this is the linear solve residual
		For "steady" solve, this is the change in solution between outer iterations
		Note that output is ORDER OF MAGNITUDE of residual norm (i.e. 1e-X, where X is the order of magnitude)
		"""

		# TODO: do some decent formatting on output

		if (solver.timeIntegrator.runSteady):
			if (solver.timeIntegrator.subiter == 1):
				res = self.solHistPrim[0] - self.solHistPrim[1]
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

		print(str(iterOut)+":\tL2: "+str(resOutL2)+", \tL1:"+str(resOutL1))

		self.resNormL2 = resNormL2
		self.resNormL1 = resNormL1
		self.resNormHistory[solver.timeIntegrator.iter-1, :] = [resNormL2, resNormL1]
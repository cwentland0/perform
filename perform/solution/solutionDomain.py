import perform.constants as const
from perform.inputFuncs import getInitialConditions, catchList, catchInput, readInputFile
from perform.timeIntegrator.explicitIntegrator import rkExplicit
from perform.timeIntegrator.implicitIntegrator import bdf
from perform.gasModel.caloricallyPerfectGas import caloricallyPerfectGas
from perform.solution.solutionPhys import solutionPhys
from perform.solution.solutionInterior import solutionInterior
from perform.solution.solutionBoundary.solutionInlet import solutionInlet 
from perform.solution.solutionBoundary.solutionOutlet import solutionOutlet
from perform.spaceSchemes import calcRHS
from perform.Jacobians import calcDResDSolPrim

import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import pdb

class solutionDomain:
	"""
	Container class for interior and boundary physical solutions
	"""

	def __init__(self, solver):

		paramDict = solver.paramDict

		# time integrator
		if (solver.timeScheme == "bdf"):
			self.timeIntegrator = bdf(paramDict)
		elif (solver.timeScheme == "rkExp"):
			self.timeIntegrator = rkExplicit(paramDict)
		else:
			raise ValueError("Invalid choice of timeScheme: "+solver.timeScheme)

		# gas model
		gasFile = str(paramDict["gasFile"]) 		# gas properties file (string)
		gasDict = readInputFile(gasFile) 
		gasType = catchInput(gasDict, "gasType", "cpg")
		if (gasType == "cpg"):
			self.gasModel = caloricallyPerfectGas(gasDict)
		else:
			raise ValueError("Ivalid choice of gasType: " + gasType)
		gas = self.gasModel

		# solution
		solPrim0    = getInitialConditions(self, solver)
		self.solInt = solutionInterior(gas, solPrim0, solver, self.timeIntegrator)
		self.solIn  = solutionInlet(gas, solver)
		self.solOut = solutionOutlet(gas, solver)

		# average solution for Roe scheme
		if (solver.spaceScheme == "roe"):
			onesProf    = np.ones((self.gasModel.numEqs, self.solInt.numCells+1), dtype=const.realType)
			self.solAve  = solutionPhys(gas, onesProf, self.solInt.numCells+1)

		# for flux calculations
		onesProf = np.ones((self.gasModel.numEqs, self.solInt.numCells+1), dtype=const.realType)
		self.solL = solutionPhys(gas, onesProf, self.solInt.numCells+1)
		self.solR = solutionPhys(gas, onesProf, self.solInt.numCells+1)

		# to avoid repeated concatenation of ghost cell states
		self.solPrimFull = np.zeros((self.gasModel.numEqs, self.solIn.numCells+self.solInt.numCells+self.solOut.numCells), dtype=const.realType)
		self.solConsFull = np.zeros(self.solPrimFull.shape, dtype=const.realType)

		# probe storage (as this can include boundaries as well)
		self.probeLocs 		= catchList(paramDict, "probeLocs", [None])
		self.probeVars 		= catchList(paramDict, "probeVars", [None])
		if ((self.probeLocs[0] is not None) and (self.probeVars[0] is not None)): 
			self.numProbes 		= len(self.probeLocs)
			self.numProbeVars 	= len(self.probeVars)
			self.probeVals 		= np.zeros((self.numProbes, self.numProbeVars, solver.numSteps), dtype=const.realType)

			# get probe locations
			self.probeIdxs = [None] * self.numProbes
			self.probeSecs = [None] * self.numProbes
			for idx, probeLoc in enumerate(self.probeLocs):
				if (probeLoc > solver.mesh.xR):
					self.probeSecs[idx] = "outlet"
				elif (probeLoc < solver.mesh.xL):
					self.probeSecs[idx] = "inlet"
				else:
					self.probeSecs[idx] = "interior"
					self.probeIdxs[idx] = np.absolute(solver.mesh.xCell - probeLoc).argmin()

			assert (not ((("outlet" in self.probeSecs) or ("inlet" in self.probeSecs)) 
						and (("source" in self.probeVars) or ("rhs" in self.probeVars))) ), \
						"Cannot probe source or RHS in inlet/outlet"

		else:
			self.numProbes = 0

		# copy this for use with plotting functions
		solver.numProbes = self.numProbes
		solver.probeVars = self.probeVars

		# TODO: include initial conditions in probeVals, timeVals
		self.timeVals = np.linspace(solver.dt * (solver.timeIter),
									solver.dt * (solver.timeIter - 1 + solver.numSteps),
								 	solver.numSteps, dtype = const.realType)


		# for compatability with hyper-reduction
		# are overwritten if actually using hyper-reduction
		self.numSampCells     = solver.mesh.numCells
		self.numFluxFaces     = solver.mesh.numCells + 1
		self.numGradCells     = solver.mesh.numCells
		self.directSampIdxs   = np.arange(0, solver.mesh.numCells)
		self.fluxSampLIdxs 	  = np.arange(0, solver.mesh.numCells+1)
		self.fluxSampRIdxs 	  = np.arange(1, solver.mesh.numCells+2)
		self.gradIdxs         = np.arange(1, solver.mesh.numCells+1)
		self.gradNeighIdxs    = np.arange(0, solver.mesh.numCells+2)
		self.gradNeighExtract = np.arange(1, solver.mesh.numCells+1)
		self.fluxLExtract     = np.arange(1, solver.mesh.numCells+1)
		self.fluxRExtract     = np.arange(0, solver.mesh.numCells)
		self.gradLExtract     = np.arange(0, solver.mesh.numCells)
		self.gradRExtract     = np.arange(0, solver.mesh.numCells)
		self.fluxRHSIdxs      = np.arange(0, solver.mesh.numCells)


	def fillSolFull(self):
		"""
		Fill solPrimFull and solConsFull from interior and ghost cells
		"""

		solInt = self.solInt
		solIn  = self.solIn
		solOut = self.solOut

		idxIn = solIn.numCells
		idxInt = idxIn + solInt.numCells

		# solPrimFull
		self.solPrimFull[:,:idxIn] 		 = solIn.solPrim.copy()
		self.solPrimFull[:,idxIn:idxInt] = solInt.solPrim.copy()
		self.solPrimFull[:,idxInt:] 	 = solOut.solPrim.copy()

		# solConsFull
		self.solConsFull[:,:idxIn] 		 = solIn.solCons.copy()
		self.solConsFull[:,idxIn:idxInt] = solInt.solCons.copy()
		self.solConsFull[:,idxInt:] 	 = solOut.solCons.copy()


	def advanceIter(self, solver):
		"""
		Advance physical solution forward one time iteration
		"""

		if (not solver.runSteady): print("Iteration "+str(solver.iter))

		for self.timeIntegrator.subiter in range(1, self.timeIntegrator.subiterMax+1):
				
			self.advanceSubiter(solver)

			# iterative solver convergence
			if (self.timeIntegrator.timeType == "implicit"):
				self.solInt.calcResNorms(solver, self.timeIntegrator.subiter)
				if (self.solInt.resNormL2 < self.timeIntegrator.resTol): break

		# "steady" convergence
		if solver.runSteady:
			self.solInt.calcDSolNorms(solver, self.timeIntegrator.timeType)

		self.solInt.updateSolHist() 


	def advanceSubiter(self, solver):
		"""
		Advance physical solution forward one subiteration of time integrator
		"""

		calcRHS(self, solver)

		solInt = self.solInt

		if (self.timeIntegrator.timeType == "implicit"):

			solInt.res = self.timeIntegrator.calcResidual(solInt.solHistCons, solInt.RHS, solver)
			resJacob = calcDResDSolPrim(self, solver)

			dSol = spsolve(resJacob, solInt.res.ravel('F'))
			
			# if solving in dual time, solving for primitive state
			if (self.timeIntegrator.dualTime):
				solInt.solPrim += dSol.reshape((self.gasModel.numEqs, solver.mesh.numCells), order='F')
			else:
				solInt.solCons += dSol.reshape((self.gasModel.numEqs, solver.mesh.numCells), order='F')
				
			solInt.updateState(fromCons = (not self.timeIntegrator.dualTime))
			solInt.solHistCons[0] = solInt.solCons.copy() 
			solInt.solHistPrim[0] = solInt.solPrim.copy() 

			# borrow solInt.res to store linear solve residual	
			res = resJacob @ dSol - solInt.res.ravel('F')
			solInt.res = np.reshape(res, (self.gasModel.numEqs, solver.mesh.numCells), order='F')

		else:

			# pdb.set_trace()
			dSol = self.timeIntegrator.solveSolChange(solInt.RHS)
			solInt.solCons = solInt.solHistCons[0] + dSol
			solInt.updateState(fromCons=True)


	def calcBoundaryCells(self, solver):
		"""
		Helper function to update boundary ghost cells
		"""

		self.solIn.calcBoundaryState(solver, solPrim=self.solInt.solPrim, solCons=self.solInt.solCons)
		self.solOut.calcBoundaryState(solver, solPrim=self.solInt.solPrim, solCons=self.solInt.solCons)


	def writeIterOutputs(self, solver):
		"""
		Helper function to save restart files and update probe/snapshot data
		"""

		# write restart files
		if (solver.saveRestarts and ((solver.iter % solver.restartInterval) == 0)): 
			self.solInt.writeRestartFile(solver)	 

		# update probe data
		if (self.numProbes > 0): 
			self.updateProbes(solver)

		# update snapshot data (not written if running steady)
		if (not solver.runSteady):
			if (( solver.iter % solver.outInterval) == 0):
				self.solInt.updateSnapshots(solver)


	def writeSteadyOutputs(self, solver):
		"""
		Helper function for write "steady" outputs and check "convergence" criterion
		"""

		# update convergence and field data file on disk
		if ((solver.iter % solver.outInterval) == 0): 
			self.solInt.writeSteadyData(solver)

		# check for "convergence"
		breakFlag = False
		if (self.solInt.dSolNormL2 < solver.steadyTol): 
			print("Steady solution criterion met, terminating run")
			breakFlag = True

		return breakFlag


	def writeFinalOutputs(self, solver):
		"""
		Helper function to write final field and probe data to disk
		"""

		if solver.solveFailed: solver.simType += "_FAILED"

		if (not solver.runSteady):		
			self.solInt.writeSnapshots(solver)
		
		if (self.numProbes > 0):
			self.writeProbes(solver)


	def updateProbes(self, solver):
		"""
		Update probe storage
		"""

		# TODO: throw error for source probe in ghost cells

		for probeIter, probeIdx in enumerate(self.probeIdxs):

			probeSec = self.probeSecs[probeIter]
			if (probeSec == "inlet"):
				solPrimProbe = self.solIn.solPrim[:,0]
				solConsProbe = self.solIn.solCons[:,0]
			elif (probeSec == "outlet"):
				solPrimProbe = self.solOut.solPrim[:,0]
				solConsProbe = self.solOut.solCons[:,0]
			else:
				solPrimProbe = self.solInt.solPrim[:,probeIdx]
				solConsProbe = self.solInt.solCons[:,probeIdx]
				solSourceProbe = self.solInt.source[:,probeIdx]

			try:
				probe = []
				for varStr in self.probeVars:
					if (varStr == "pressure"):
						probe.append(solPrimProbe[0])
					elif (varStr == "velocity"):
						probe.append(solPrimProbe[1])
					elif (varStr == "temperature"):
						probe.append(solPrimProbe[2])
					elif (varStr == "source"):
						probe.append(solSourceProbe[0])
					elif (varStr == "density"):
						probe.append(solConsProbe[0])
					elif (varStr == "momentum"):
						probe.append(solConsProbe[1])
					elif (varStr == "energy"):
						probe.append(solConsProbe[2])
					elif (varStr == "species"):
						probe.append(solPrimProbe[3])
					elif (varStr[:7] == "species"):
						specIdx = int(varStr[7:])
						probe.append(solPrimProbe[3+specIdx-1])
					elif (varStr[:15] == "density-species"):
						specIdx = int(varStr[15:])
						probe.append(solConsProbe[3+specIdx-1])
			except:
				raise ValueError("Invalid probe variable "+str(varStr))
			
			self.probeVals[probeIter, :, solver.iter-1] = probe
		

	def writeProbes(self, solver):
		"""
		Save probe data to disk
		"""

		probeFileBaseName = "probe"
		for visVar in self.probeVars:
			probeFileBaseName += "_"+visVar		

		for probeNum in range(self.numProbes):

			# account for failed simulations
			timeOut  = self.timeVals[:solver.iter]
			probeOut = self.probeVals[probeNum,:,:solver.iter]

			probeFileName = probeFileBaseName + "_" + str(probeNum+1) + "_" + solver.simType + ".npy"
			probeFile = os.path.join(const.probeOutputDir, probeFileName)

			probeSave = np.concatenate((timeOut[None,:], probeOut), axis=0) 
			np.save(probeFile, probeSave)
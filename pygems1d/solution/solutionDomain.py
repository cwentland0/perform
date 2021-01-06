import pygems1d.constants as const
from pygems1d.inputFuncs import getInitialConditions, catchList
from pygems1d.solution.solutionInterior import solutionInterior
from pygems1d.solution.solutionBoundary.solutionInlet import solutionInlet 
from pygems1d.solution.solutionBoundary.solutionOutlet import solutionOutlet

import os
import numpy as np
import pdb

class solutionDomain:
	"""
	Container class for interior and boundary physical solutions
	"""

	def __init__(self, solver):

		timeInt = solver.timeIntegrator

		# solution
		solPrim0, solCons0 	= getInitialConditions(solver)
		self.solInt 		= solutionInterior(solPrim0, solCons0, solver)
		self.solIn 			= solutionInlet(solver)
		self.solOut 		= solutionOutlet(solver)

		# probe storage (as this can include boundaries as well)
		paramDict 			= solver.paramDict
		self.probeLocs 		= catchList(paramDict, "probeLocs", [None])
		self.probeVars 		= catchList(paramDict, "probeVars", [None])
		if ((self.probeLocs[0] is not None) and (self.probeVars[0] is not None)): 
			self.numProbes 		= len(self.probeLocs)
			self.numProbeVars 	= len(self.probeVars)
			self.probeVals 		= np.zeros((self.numProbes, self.numProbeVars, timeInt.numSteps), dtype=const.realType)

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
		self.timeVals = np.linspace(timeInt.dt * (timeInt.timeIter),
								 	timeInt.dt * (timeInt.timeIter + solver.timeIntegrator.numSteps), 
								 	timeInt.numSteps, dtype = const.realType)



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
		if (solver.saveRestarts and ((solver.timeIntegrator.iter % solver.restartInterval) == 0)): 
			self.solInt.writeRestartFile(solver)	 

		# update probe data
		if (self.numProbes > 0): 
			self.updateProbes(solver)

		# update snapshot data (not written if running steady)
		if (not solver.timeIntegrator.runSteady):
			if (( solver.timeIntegrator.iter % solver.outInterval) == 0):
				self.solInt.updateSnapshots(solver)


	def writeSteadyOutputs(self, solver):
		"""
		Helper function for write "steady" outputs and check "convergence" criterion
		"""

		# update convergence and field data file on disk
		if ((solver.timeIntegrator.iter % solver.outInterval) == 0): 
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

		if (not solver.timeIntegrator.runSteady):		
			self.solInt.writeSnapshots(solver)
		
		if (self.numProbes > 0):
			self.writeProbes(solver)


	def updateProbes(self, solver):
		"""
		Update probe storage
		"""

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
			
			self.probeVals[probeIter, :, solver.timeIntegrator.iter-1] = probe
		

	def writeProbes(self, solver):
		"""
		Save probe data to disk
		"""

		probeFileBaseName = "probe"
		for visVar in self.probeVars:
			probeFileBaseName += "_"+visVar		

		for probeNum in range(self.numProbes):

			# account for failed simulations
			timeOut  = self.timeVals[:solver.timeIntegrator.iter]
			probeOut = self.probeVals[probeNum,:,:solver.timeIntegrator.iter]

			probeFileName = probeFileBaseName + "_" + str(probeNum+1) + "_" + solver.simType + ".npy"
			probeFile = os.path.join(const.probeOutputDir, probeFileName)

			probeSave = np.concatenate((timeOut[:,None], probeOut.T), axis=1) 
			np.save(probeFile, probeSave)
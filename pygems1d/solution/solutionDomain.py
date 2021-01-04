import pygems1d.constants as const
from pygems1d.inputFuncs import getInitialConditions
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

		solPrim0, solCons0 	= getInitialConditions(solver)
		self.solInt 		= solutionInterior(solPrim0, solCons0, solver)
		self.solIn 			= solutionInlet(solver)
		self.solOut 		= solutionOutlet(solver)

	def calcBoundaryCells(self, solver):
		"""
		Helper function to update boundary ghost cells
		"""

		self.solIn.calcBoundaryState(solver, solPrim=self.solInt.solPrim, solCons=self.solInt.solCons)
		self.solOut.calcBoundaryState(solver, solPrim=self.solInt.solPrim, solCons=self.solInt.solCons)

	def writeOutputs(self, solver):
		"""
		Helper function to save/write restart files, probe data, and unsteady field data
		"""

		# write restart files
		if (solver.saveRestarts and ((solver.timeIntegrator.iter % solver.restartInterval) == 0)): 
			self.writeRestartFile(solver)	 


		# update probe data


		# update snapshot data

	def writeRestartFile(self, solver):
		"""
		Write restart files containing primitive and conservative fields, plus physical time 
		"""

		# TODO: write previous time step(s) for multi-step methods, to preserve time accuracy at restart

		# write restart file to zipped file
		restartFile = os.path.join(const.restartOutputDir, "restartFile_"+str(solver.restartIter)+".npz")
		np.savez(restartFile, solTime = solver.solTime, solPrim = self.solInt.solPrim, solCons = self.solInt.solCons)

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
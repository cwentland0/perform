import pygems1d.constants
from pygems1d.inputFuncs import catchInput

import numpy as np
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
		self.subiterMax = None	# maximum number of subiterations for multi-stage explicit or iterative schemes

	def advanceIter(self, solDomain, solROM, solver):

		if (not solver.timeIntegrator.runSteady): print("Iteration "+str(self.iter))

		for self.subiter in range(1, self.subiterMax+1):	
			self.advanceSubiter(solDomain, solROM, solver)

			# iterative solver convergence
			if (self.timeType == "implicit"):
				solDomain.solInt.calcResNorms(solver)
				if (solDomain.solInt.resNormL2 < self.resTol): break

		# "steady" convergence
		if self.runSteady:
			solDomain.solInt.calcDSolNorms(solver)

		self.timeIter += 1
		solver.solTime += solver.timeIntegrator.dt
		solDomain.solInt.updateSolHist(solver) 
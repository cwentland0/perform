from pygems1d.inputFuncs import getInitialConditions
from pygems1d.solution.solutionInterior import solutionInterior
from pygems1d.solution.solutionBoundary.solutionInlet import solutionInlet 
from pygems1d.solution.solutionBoundary.solutionOutlet import solutionOutlet

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
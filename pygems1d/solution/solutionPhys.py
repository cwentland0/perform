from pygems1d.constants import realType
from pygems1d.stateFuncs import calcStateFromPrim, calcStateFromCons

import numpy as np
import pdb

# TODO error checking on initial solution load

class solutionPhys:
	"""
	Base class for physical solution (opposed to ROM solution)
	"""

	def __init__(self, solDomain, solPrimIn, solConsIn, numCells, solver):
		
		self.gasModel = solDomain.gasModel

		self.numCells = numCells

		# solution and mixture properties
		self.solPrim	= np.zeros((self.gasModel.numEqs, numCells), dtype=realType)		# solution in primitive variables
		self.solCons	= np.zeros((self.gasModel.numEqs, numCells), dtype=realType)		# solution in conservative variables
		self.mwMix 		= np.zeros(numCells, dtype=realType)					# mixture molecular weight
		self.RMix		= np.zeros(numCells, dtype=realType)					# mixture specific gas constant
		self.gammaMix 	= np.zeros(numCells, dtype=realType)					# mixture ratio of specific heats
		self.enthRefMix = np.zeros(numCells, dtype=realType)					# mixture reference enthalpy
		self.CpMix 		= np.zeros(numCells, dtype=realType)					# mixture specific heat at constant pressure

		# load initial condition and check size
		assert(solPrimIn.shape == (self.gasModel.numEqs, numCells))
		assert(solConsIn.shape == (self.gasModel.numEqs, numCells))
		self.solPrim = solPrimIn.copy()
		self.solCons = solConsIn.copy()


	def updateState(self, fromCons=True):
		"""
		Update state and mixture gas properties
		"""

		if fromCons:
			self.solPrim, self.RMix, self.enthRefMix, self.CpMix = calcStateFromCons(self.solCons, self.gasModel)
		else:
			self.solCons, self.RMix, self.enthRefMix, self.CpMix = calcStateFromPrim(self.solPrim, self.gasModel)
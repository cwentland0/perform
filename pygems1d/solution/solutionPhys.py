from pygems1d.constants import realType
from pygems1d.stateFuncs import calcStateFromPrim, calcStateFromCons

import numpy as np
import pdb

# TODO error checking on initial solution load

class solutionPhys:
	"""
	Base class for physical solution (opposed to ROM solution)
	"""

	def __init__(self, solPrimIn, solConsIn, numCells, solver):
		
		gas 		= solver.gasModel
		timeInt 	= solver.timeIntegrator

		# solution and mixture properties
		self.solPrim	= np.zeros((numCells, gas.numEqs), dtype=realType)		# solution in primitive variables
		self.solCons	= np.zeros((numCells, gas.numEqs), dtype=realType)		# solution in conservative variables
		self.mwMix 		= np.zeros(numCells, dtype=realType)					# mixture molecular weight
		self.RMix		= np.zeros(numCells, dtype=realType)					# mixture specific gas constant
		self.gammaMix 	= np.zeros(numCells, dtype=realType)					# mixture ratio of specific heats
		self.enthRefMix = np.zeros(numCells, dtype=realType)					# mixture reference enthalpy
		self.CpMix 		= np.zeros(numCells, dtype=realType)					# mixture specific heat at constant pressure

		# load initial condition and check size
		assert(solPrimIn.shape == (numCells, gas.numEqs))
		assert(solConsIn.shape == (numCells, gas.numEqs))
		self.solPrim = solPrimIn.copy()
		self.solCons = solConsIn.copy()


	def updateState(self, gas, fromCons=True):
		"""
		Update state and mixture gas properties
		"""

		if fromCons:
			self.solPrim, self.RMix, self.enthRefMix, self.CpMix = calcStateFromCons(self.solCons, gas)
		else:
			self.solCons, self.RMix, self.enthRefMix, self.CpMix = calcStateFromPrim(self.solPrim, gas)
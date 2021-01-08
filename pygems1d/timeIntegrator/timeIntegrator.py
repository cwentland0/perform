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
		self.timeScheme = str(paramDict["timeScheme"])	# time integration scheme
		self.timeOrder 	= int(paramDict["timeOrder"])	# time integration order of accuracy
		assert (self.timeOrder >= 1), "timeOrder only accepts positive integer values."

		self.subiter 	= 1		# subiteration number for multi-stage schemes
		self.subiterMax = None	# maximum number of subiterations for multi-stage explicit or iterative schemes



class romModel:
	"""
	Base class for ROM model
	"""

	def __init__(self, romMethod, latentDim):

		self.romMethod = romMethod
		self.latentDim = latentDim

		
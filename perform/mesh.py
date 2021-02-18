from perform.constants import realType

import numpy as np

class mesh:

	def __init__(self, meshDict):
		self.xL 		= float(meshDict["xL"])
		self.xR 		= float(meshDict["xR"])
		self.numCells 	= int(meshDict["numCells"])
		self.numFaces 	= self.numCells + 1
		self.xFace 		= np.linspace(self.xL, self.xR, self.numCells + 1, dtype=realType)
		self.xCell 		= (self.xFace[1:] + self.xFace[:-1]) / 2.0

		# TODO: this should be an array when the non-uniform mesh is implemented
		# TODO: also need distances between cell-centers and faces for viscous flux calculations
		self.dx 		= self.xFace[1] - self.xFace[0]


# TODO: uniform, non-uniform meshes, accompanying gradient operators and face reconstructions
import numpy as np

from perform.constants import REAL_TYPE

# TODO: uniform, non-uniform meshes, grad operators and face reconstruction


class Mesh:

	def __init__(self, meshDict):
		self.x_left = float(meshDict["x_left"])
		self.x_right = float(meshDict["x_right"])
		self.num_cells = int(meshDict["num_cells"])
		self.num_faces = self.num_cells + 1
		self.x_face = np.linspace(self.x_left, self.x_right,
									self.num_cells + 1, dtype=REAL_TYPE)
		self.x_cell = (self.x_face[1:] + self.x_face[:-1]) / 2.0

		# TODO: Should be an array for non-uniform mesh
		# TODO: Need distances between cell-centers and faces for visc flux
		self.dx = self.x_face[1] - self.x_face[0]

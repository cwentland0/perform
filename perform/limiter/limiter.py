import numpy as np


class Limiter():
	"""
	Base class for gradient limiters
	"""

	def __init__(self):

		pass

	def calc_neighbor_minmax(self, sol):
		"""
		Find minimum and maximum of cell state and neighbor cell state
		"""

		# max and min of cell and neighbors
		sol_max = sol.copy()
		sol_min = sol.copy()

		# first compare against right neighbor
		sol_max[:, :-1] = np.maximum(sol[:, :-1], sol[:, 1:])
		sol_min[:, :-1] = np.minimum(sol[:, :-1], sol[:, 1:])

		# then compare agains left neighbor
		sol_max[:, 1:] = np.maximum(sol_max[:, 1:], sol[:, :-1])
		sol_min[:, 1:] = np.minimum(sol_min[:, 1:], sol[:, :-1])

		return sol_min, sol_max

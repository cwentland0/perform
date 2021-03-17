import numpy as np

from perform.constants import REAL_TYPE
from perform.limiter.limiter import Limiter


class BarthJespLimiter(Limiter):
	"""
	Barth-Jespersen limiter
	Ensures that no new minima or maxima are introduced in reconstruction
	"""

	def __init__(self):

		super().__init__()

	def calc_limiter(self, sol_domain, grad):
		"""
		Compute multiplicative limiter
		"""

		sol_prim = sol_domain.sol_prim_full[:, sol_domain.grad_idxs]

		# get min/max of cell and neighbors
		sol_prim_min, sol_prim_max = \
			self.calc_neighbor_minmax(sol_domain.sol_prim_full[:, sol_domain.grad_neigh_idxs])

		# extract gradient cells
		sol_prim_min = sol_prim_min[:, sol_domain.grad_neigh_extract]
		sol_prim_max = sol_prim_max[:, sol_domain.grad_neigh_extract]

		# unconstrained reconstruction at neighboring cell centers
		d_sol_prim = grad * sol_domain.mesh.dx
		sol_prim_left = sol_prim - d_sol_prim
		sol_prim_right = sol_prim + d_sol_prim

		# limiter defaults to 1
		phi_left = np.ones(sol_prim.shape, dtype=REAL_TYPE)
		phi_right = np.ones(sol_prim.shape, dtype=REAL_TYPE)

		# find idxs where difference is either positive or negative
		cond1_left = ((sol_prim_left - sol_prim) > 0)
		cond1_right = ((sol_prim_right - sol_prim) > 0)
		cond2_left = ((sol_prim_left - sol_prim) < 0)
		cond2_right = ((sol_prim_right - sol_prim) < 0)

		# threshold limiter for left and right reconstruction
		phi_left[cond1_left] = np.minimum(1.0,
			(sol_prim_max[cond1_left] - sol_prim[cond1_left])
			/ (sol_prim_left[cond1_left] - sol_prim[cond1_left]))
		phi_right[cond1_right] = np.minimum(1.0,
			(sol_prim_max[cond1_right] - sol_prim[cond1_right])
			/ (sol_prim_right[cond1_right] - sol_prim[cond1_right]))
		phi_left[cond2_left] = np.minimum(1.0,
			(sol_prim_min[cond2_left] - sol_prim[cond2_left])
			/ (sol_prim_left[cond2_left] - sol_prim[cond2_left]))
		phi_right[cond2_right] = np.minimum(1.0,
			(sol_prim_min[cond2_right] - sol_prim[cond2_right])
			/ (sol_prim_right[cond2_right] - sol_prim[cond2_right]))

		# take minimum limiter from left and right
		phi = np.minimum(phi_left, phi_right)

		return phi

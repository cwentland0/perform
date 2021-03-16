import numpy as np

from perform.constants import REAL_TYPE


def calc_cell_gradients(sol_domain, solver):
	"""
	Compute cell-centered gradients for higher-order face reconstructions
	Also calculate gradient limiters if requested
	"""

	# Compute gradients via finite difference stencil
	sol_prim_grad = np.zeros((sol_domain.gas_model.num_eqs,
								sol_domain.num_grad_cells), dtype=REAL_TYPE)
	if solver.space_order == 2:
		sol_prim_grad = ((0.5 / solver.mesh.dx)
			* (sol_domain.sol_prim_full[:, sol_domain.grad_idxs + 1]
			- sol_domain.sol_prim_full[:, sol_domain.grad_idxs - 1]))
	else:
		raise ValueError("Order " + str(solver.space_order)
							+ " gradient calculations not implemented")

	# compute gradient limiter and limit gradient, if requested
	if solver.grad_limiter != "":

		# Barth-Jespersen
		if solver.grad_limiter == "barth":
			phi = limiter_barth_jespersen(sol_domain, sol_prim_grad, solver.mesh)

		# Venkatakrishnan, no correction
		elif solver.grad_limiter == "venkat":
			phi = limiter_venkatakrishnan(sol_domain, sol_prim_grad, solver.mesh)

		else:
			raise ValueError("Invalid input for grad_limiter: "
								+ str(solver.grad_limiter))

		# limit gradient
		sol_prim_grad = sol_prim_grad * phi

	return sol_prim_grad


def find_neighbor_minmax(sol):
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


def limiter_barth_jespersen(sol_domain, grad, mesh):
	"""
	Barth-Jespersen limiter
	Ensures that no new minima or maxima are introduced in reconstruction
	"""

	sol_prim = sol_domain.sol_prim_full[:, sol_domain.grad_idxs]

	# get min/max of cell and neighbors
	sol_prim_min, sol_prim_max = \
		find_neighbor_minmax(sol_domain.sol_prim_full[:, sol_domain.grad_neigh_idxs])

	# extract gradient cells
	sol_prim_min = sol_prim_min[:, sol_domain.grad_neigh_extract]
	sol_prim_max = sol_prim_max[:, sol_domain.grad_neigh_extract]

	# unconstrained reconstruction at neighboring cell centers
	d_sol_prim = grad * mesh.dx
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


def limiter_venkatakrishnan(sol_domain, grad, mesh):
	"""
	Venkatakrishnan limiter
	Differentiable, but limits in uniform regions
	"""

	sol_prim = sol_domain.sol_prim_full[:, sol_domain.grad_idxs]

	# get min/max of cell and neighbors
	sol_prim_min, sol_prim_max = \
		find_neighbor_minmax(sol_domain.sol_prim_full[:, sol_domain.grad_neigh_idxs])

	# extract gradient cells
	sol_prim_min = sol_prim_min[:, sol_domain.grad_neigh_extract]
	sol_prim_max = sol_prim_max[:, sol_domain.grad_neigh_extract]

	# unconstrained reconstruction at neighboring cell centers
	d_sol_prim = grad * mesh.dx
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

	# (y^2 + 2y) / (y^2 + y + 2)
	def venkat_function(maxVals, cell_vals, face_vals):
		frac = (maxVals - cell_vals) / (face_vals - cell_vals)
		frac_sq = np.square(frac)
		venk_vals = (frac_sq + 2.0 * frac) / (frac_sq + frac + 2.0)
		return venk_vals

	# apply smooth Venkatakrishnan function
	phi_left[cond1_left] = \
		venkat_function(sol_prim_max[cond1_left],
						sol_prim[cond1_left],
						sol_prim_left[cond1_left])
	phi_right[cond1_right] = \
		venkat_function(sol_prim_max[cond1_right],
						sol_prim[cond1_right],
						sol_prim_right[cond1_right])
	phi_left[cond2_left] = \
		venkat_function(sol_prim_min[cond2_left],
						sol_prim[cond2_left],
						sol_prim_left[cond2_left])
	phi_right[cond2_right] = \
		venkat_function(sol_prim_min[cond2_right],
						sol_prim[cond2_right],
						sol_prim_right[cond2_right])

	# take minimum limiter from left and right
	phi = np.minimum(phi_left, phi_right)

	return phi

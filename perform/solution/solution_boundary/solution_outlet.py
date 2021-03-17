from math import pow, sqrt

from perform.solution.solution_boundary.solution_boundary import SolutionBoundary


class SolutionOutlet(SolutionBoundary):
	"""
	Outlet ghost cell solution
	"""

	def __init__(self, gas, solver):

		param_dict = solver.param_dict
		self.bound_cond = param_dict["bound_cond_outlet"]

		# add assertions to check that required properties are specified
		if self.bound_cond == "subsonic":
			self.bound_func = self.calc_subsonic_bc
		elif self.bound_cond == "meanflow":
			self.bound_func = self.calc_mean_flow_bc
		else:
			raise ValueError("Invalid outlet boundary condition selection: "
							+ str(self.bound_cond))

		super().__init__(gas, solver, "outlet")

	def calc_subsonic_bc(self, sol_time, space_order, sol_prim=None, sol_cons=None):
		"""
		Subsonic outflow, specify outlet static pressure
		"""

		# TODO: deal with mass fraction at outlet, should not be fixed

		assert ((sol_prim is not None) and (sol_cons is not None)), \
			"Must provide primitive and conservative interior state."

		press_bound = self.press
		if self.pert_type == "pressure":
			press_bound *= (1.0 + self.calc_pert(sol_time))

		# chemical composition assumed constant near boundary
		r_mix = self.r_mix[0]
		gamma_mix = self.gamma_mix[0]
		gamma_mix_m1 = gamma_mix - 1.0

		# calculate interior state
		press_p1 = sol_prim[0, -1]
		press_p2 = sol_prim[0, -2]
		rho_p1 = sol_cons[0, -1]
		rho_p2 = sol_cons[0, -2]
		vel_p1 = sol_prim[1, -1]
		vel_p2 = sol_prim[1, -2]

		# outgoing characteristics information
		s_p1 = press_p1 / pow(rho_p1, gamma_mix)
		s_p2 = press_p2 / pow(rho_p2, gamma_mix)
		c_p1 = sqrt(gamma_mix * r_mix * sol_prim[2, -1])
		c_p2 = sqrt(gamma_mix * r_mix * sol_prim[2, -2])
		j_p1 = vel_p1 + 2.0 * c_p1 / gamma_mix_m1
		j_p2 = vel_p2 + 2.0 * c_p2 / gamma_mix_m1

		# extrapolate to exterior
		if space_order == 1:
			s = s_p1
			j = j_p1
		elif space_order == 2:
			s = 2.0 * s_p1 - s_p2
			j = 2.0 * j_p1 - j_p2
		else:
			raise ValueError("Higher order extrapolation implementation "
							+ "required for spatial order " + str(space_order))

		# compute exterior state
		self.sol_prim[0, 0] = press_bound
		rho_bound = pow((press_bound / s), (1.0 / gamma_mix))
		c_bound = sqrt(gamma_mix * press_bound / rho_bound)
		self.sol_prim[1, 0] = j - 2.0 * c_bound / gamma_mix_m1
		self.sol_prim[2, 0] = press_bound / (r_mix * rho_bound)

	def calc_mean_flow_bc(self, sol_time, space_order, sol_prim=None, sol_cons=None):
		"""
		Non-reflective boundary, unsteady solution is
		perturbation about mean flow solution
		Refer to documentation for derivation
		"""

		assert (sol_prim is not None), "Must provide primitive interior state"

		rho_c_mean = self.vel
		rho_cp_mean = self.rho
		press_back = self.press

		if self.pert_type == "pressure":
			press_back *= (1.0 + self.calc_pert(sol_time))

		# interior quantities
		press_out = sol_prim[0, -2:]
		vel_out = sol_prim[1, -2:]
		temp_out = sol_prim[2, -2:]
		mass_frac_out = sol_prim[3:, -2:]

		# characteristic variables
		w_1_out = temp_out - press_out / rho_cp_mean
		w_2_out = vel_out + press_out / rho_c_mean
		w_4_out = mass_frac_out

		# extrapolate to exterior
		if space_order == 1:
			w_1_bound = w_1_out[0]
			w_2_bound = w_2_out[0]
			w_4_bound = w_4_out[:, 0]
		elif space_order == 2:
			w_1_bound = 2.0 * w_1_out[0] - w_1_out[1]
			w_2_bound = 2.0 * w_2_out[0] - w_2_out[1]
			w_4_bound = 2.0 * w_4_out[:, 0] - w_4_out[:, 0]
		else:
			raise ValueError("Higher order extrapolation implementation "
							+ "required for spatial order " + str(space_order))

		# compute exterior state
		press_bound = (w_2_bound * rho_c_mean + press_back) / 2.0
		self.sol_prim[0, 0] = press_bound
		self.sol_prim[1, 0] = (press_bound - press_back) / rho_c_mean
		self.sol_prim[2, 0] = w_1_bound + press_bound / rho_cp_mean
		self.sol_prim[3:, 0] = w_4_bound

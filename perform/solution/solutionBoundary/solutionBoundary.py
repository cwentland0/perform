from math import sin, pi

import numpy as np

from perform.constants import REAL_TYPE
from perform.solution.solutionPhys import SolutionPhys 
from perform.inputFuncs import parse_bc


class SolutionBoundary(SolutionPhys):
	"""
	Ghost cell solution
	"""

	def __init__(self, gas, solver, bound_type):

		param_dict = solver.param_dict

		# this generally stores fixed/stagnation properties
		self.press, self.vel, self.temp, self.mass_fracs, self.rho, \
			self.pert_type, self.pert_perc, self.pert_freq = parse_bc(bound_type, param_dict)

		assert (len(self.mass_fracs) == gas.num_species_full), "Must provide mass fraction state for all species at boundary"
		assert (np.sum(self.mass_fracs) == 1.0), "Boundary mass fractions must sum to 1.0"

		self.cp_mix       = gas.calc_mix_cp(self.mass_fracs[gas.mass_frac_slice, None])
		self.r_mix        = gas.calc_mix_gas_constant(self.mass_fracs[gas.mass_frac_slice, None])
		self.gamma_mix    = gas.calc_mix_gamma(self.r_mix, self.cp_mix)
		self.enth_ref_mix = gas.calc_mix_enth_ref(self.mass_fracs[gas.mass_frac_slice, None])

		# this will be updated at each iteration, just initializing now
		# TODO: number of ghost cells should not always be one
		sol_dummy      = np.zeros((gas.num_eqs,1), dtype=REAL_TYPE)
		sol_dummy[0,0] = 1e6
		sol_dummy[1,0] = 1.0
		sol_dummy[2,0] = 300.0
		sol_dummy[3,0] = 1.0
		super().__init__(gas, 1, sol_prim_in=sol_dummy)
		self.sol_prim[3:,0] = self.mass_fracs[gas.mass_frac_slice]

	@profile
	def calc_pert(self, t):
		"""
		Compute sinusoidal perturbation factor 
		"""

		# TODO: add phase offset

		pert = 0.0
		for f in self.pert_freq:
			pert += sin(2.0 * pi * self.pert_freq * t)
		pert *= self.pert_perc 

		return pert

	def calc_boundary_state(self, solver, sol_prim=None, sol_cons=None):
		"""
		Run boundary calculation and update ghost cell state
		Assumed that boundary function sets primitive state
		"""

		self.bound_func(solver, sol_prim, sol_cons)
		self.update_state(from_cons = False)
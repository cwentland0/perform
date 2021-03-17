import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.flux import Flux

class InviscFlux(Flux):
	"""
	Base class for any inviscid flux scheme
	"""

	def __init__(self):

		pass

	def calc_d_inv_flux_d_sol_prim(self, sol):
		"""
		Compute Jacobian of inviscid flux vector with respect to primitive state
		Here, sol should be the solutionPhys associated with a face state
		"""

		gas = sol.gas_model

		d_flux_d_sol_prim = np.zeros((gas.num_eqs, gas.num_eqs, sol.num_cells), dtype=REAL_TYPE)

		# for convenience
		rho = sol.sol_cons[0, :]
		press = sol.sol_prim[0, :]
		vel = sol.sol_prim[1, :]
		vel_sq = np.square(vel)
		temp = sol.sol_prim[2, :]
		mass_fracs = sol.sol_prim[3:, :]
		h0 = sol.h0

		d_rho_d_press, d_rho_d_temp, d_rho_d_mass_frac = \
			gas.calc_dens_derivs(rho,
									wrt_press=True, pressure=press,
									wrt_temp=True, temperature=temp,
									wrt_spec=True, mass_fracs=sol.mass_fracs_full)

		d_enth_d_press, d_enth_d_temp, d_enth_d_mass_frac = \
			gas.calc_stag_enth_derivs(wrt_press=True,
										wrt_temp=True, mass_fracs=mass_fracs,
										wrt_spec=True, spec_enth=sol.hi)

		# continuity equation row
		d_flux_d_sol_prim[0, 0, :] = vel * d_rho_d_press
		d_flux_d_sol_prim[0, 1, :] = rho
		d_flux_d_sol_prim[0, 2, :] = vel * d_rho_d_temp
		d_flux_d_sol_prim[0, 3:, :] = vel[None, :] * d_rho_d_mass_frac

		# momentum equation row
		d_flux_d_sol_prim[1, 0, :] = d_rho_d_press * vel_sq + 1.0
		d_flux_d_sol_prim[1, 1, :] = 2.0 * rho * vel
		d_flux_d_sol_prim[1, 2, :] = d_rho_d_temp * vel_sq
		d_flux_d_sol_prim[1, 3:, :] = vel_sq[None, :] * d_rho_d_mass_frac

		# energy equation row
		d_flux_d_sol_prim[2, 0, :] = vel * (h0 * d_rho_d_press + rho * d_enth_d_press)
		d_flux_d_sol_prim[2, 1, :] = rho * (vel_sq + h0)
		d_flux_d_sol_prim[2, 2, :] = vel * (h0 * d_rho_d_temp + rho * d_enth_d_temp)
		d_flux_d_sol_prim[2, 3:, :] = \
			(vel[None, :] * (h0[None, :] * d_rho_d_mass_frac
			+ rho[None, :] * d_enth_d_mass_frac))

		# species transport row(s)
		d_flux_d_sol_prim[3:, 0, :] = mass_fracs * (d_rho_d_press * vel)[None, :]
		d_flux_d_sol_prim[3:, 1, :] = mass_fracs * rho[None, :]
		d_flux_d_sol_prim[3:, 2, :] = mass_fracs * (d_rho_d_temp * vel)[None, :]
		# TODO: vectorize
		for i in range(3, gas.num_eqs):
			for j in range(3, gas.num_eqs):
				d_flux_d_sol_prim[i, j, :] = \
					(vel * ((i == j) * rho
					+ mass_fracs[i - 3, :] * d_rho_d_mass_frac[j - 3, :]))

		return d_flux_d_sol_prim

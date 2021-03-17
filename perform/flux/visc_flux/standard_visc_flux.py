import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.flux import Flux

class StandardViscFlux(Flux):
	"""
	Standard viscous flux scheme with binary diffusion velocity approximation
	"""

	def __init__(self, sol_domain, solver):

		super().__init__()

	def calc_flux(self, sol_domain, solver):
		"""
		Compute flux array
		"""

		gas = sol_domain.gas_model
		mesh = solver.mesh
		sol_ave = sol_domain.sol_ave
		sol_prim_full = sol_domain.sol_prim_full

		# Compute 2nd-order state gradients at faces
		# TODO: generalize to higher orders of accuracy
		sol_prim_grad = np.zeros((gas.num_eqs + 1, sol_domain.num_flux_faces),
								dtype=REAL_TYPE)
		sol_prim_grad[:-1, :] = \
			((sol_prim_full[:, sol_domain.flux_samp_right_idxs]
			- sol_prim_full[:, sol_domain.flux_samp_left_idxs])
			/ mesh.dx)

		# Get gradient of last species for diffusion velocity term
		# TODO: maybe a sneakier way to do this?
		mass_fracs = gas.calc_all_mass_fracs(sol_prim_full[3:, :], threshold=False)
		sol_prim_grad[-1, :] = \
			((mass_fracs[-1, sol_domain.flux_samp_right_idxs]
			- mass_fracs[-1, sol_domain.flux_samp_left_idxs])
			/ mesh.dx)

		# Thermo and transport props
		mole_fracs = gas.calc_all_mole_fracs(sol_ave.mass_fracs_full,
											mix_mol_weight=sol_ave.mw_mix)
		spec_dyn_visc = gas.calc_species_dynamic_visc(sol_ave.sol_prim[2, :])
		therm_cond_mix = gas.calc_mix_thermal_cond(spec_dyn_visc=spec_dyn_visc,
													mole_fracs=mole_fracs)
		dyn_visc_mix = gas.calc_mix_dynamic_visc(spec_dyn_visc=spec_dyn_visc,
												mole_fracs=mole_fracs)
		mass_diff_mix = gas.calc_species_mass_diff_coeff(sol_ave.sol_cons[0, :],
														spec_dyn_visc=spec_dyn_visc)
		hi = gas.calc_spec_enth(sol_ave.sol_prim[2, :])

		# Copy for use later
		sol_ave.dyn_visc_mix = dyn_visc_mix
		sol_ave.therm_cond_mix = therm_cond_mix
		sol_ave.mass_diff_mix = mass_diff_mix
		sol_ave.hi = hi

		# Stress "tensor"
		tau = 4.0 / 3.0 * dyn_visc_mix * sol_prim_grad[1, :]

		# Diffusion velocity
		diff_vel = sol_ave.sol_cons[[0], :] * mass_diff_mix * sol_prim_grad[3:, :]

		# Correction velocity
		corr_vel = np.sum(diff_vel, axis=0)

		# Viscous flux
		flux_visc = np.zeros((gas.num_eqs, sol_domain.num_flux_faces),
							dtype=REAL_TYPE)
		flux_visc[1, :] += tau
		flux_visc[2, :] += (sol_ave.sol_prim[1, :] * tau
							+ therm_cond_mix * sol_prim_grad[2, :]
							+ np.sum(diff_vel * hi, axis=0))
		flux_visc[3:, :] += (diff_vel[gas.mass_frac_slice]
							- sol_ave.sol_prim[3:, :] * corr_vel[None, :])

		return flux_visc
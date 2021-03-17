import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix, dia_matrix, diags

import perform.constants as const



def calc_d_source_d_sol_prim(sol_int, dt):
	"""
	Compute source term Jacobian
	"""

	# TODO: does not account for reverse reaction, multiple reactions
	# TODO: does not account for temperature exponent in Arrhenius rate
	# TODO: need to check that this works for more than a two-species reaction

	gas = sol_int.gas_model

	d_source_d_sol_prim = np.zeros((gas.num_eqs, gas.num_eqs, sol_int.num_cells))

	rho = sol_int.sol_cons[0, :]
	press = sol_int.sol_prim[0, :]
	temp = sol_int.sol_prim[2, :]
	mass_fracs = sol_int.sol_prim[3:, :]

	d_rho_d_press = sol_int.d_rho_d_press
	d_rho_d_temp = sol_int.d_rho_d_temp
	d_rho_d_mass_frac = sol_int.d_rho_d_mass_frac
	wf = sol_int.wf

	spec_idxs = np.squeeze(np.argwhere(gas.nu_arr != 0.0))

	# TODO: not correct for multi-reaction
	wf_div_rho = (np.sum(wf[None, :] * gas.nu_arr[spec_idxs, None], axis=0)
					/ rho[None, :])

	# subtract, as activation energy already set as negative
	d_wf_d_temp = (wf_div_rho * d_rho_d_temp[None, :]
		- wf * gas.act_energy / temp**2)
	d_wf_d_press = wf_div_rho * d_rho_d_press[None, :]

	# TODO: incorrect for multi-reaction, should be [numSpec, numReac, num_cells]
	d_wf_d_mass_frac = wf_div_rho * d_rho_d_mass_frac
	for i in range(gas.num_species):
		pos_mf_idxs = np.nonzero(mass_fracs[i, :] > 0.0)[0]
		d_wf_d_mass_frac[i, pos_mf_idxs] += \
			wf[pos_mf_idxs] * gas.nu_arr[i] / mass_fracs[i, pos_mf_idxs]

	# TODO: for multi-reaction, should be a summation over the reactions here
	d_source_d_sol_prim[3:, 0, :] = \
		-gas.mol_weight_nu[gas.mass_frac_slice, None] * d_wf_d_press
	d_source_d_sol_prim[3:, 2, :] = \
		-gas.mol_weight_nu[gas.mass_frac_slice, None] * d_wf_d_temp

	# TODO: this is totally wrong for multi-reaction
	for i in range(gas.num_species):
		d_source_d_sol_prim[3:, 3 + i, :] = \
			-gas.mol_weight_nu[[i], None] * d_wf_d_mass_frac[[i], :]

	return d_source_d_sol_prim

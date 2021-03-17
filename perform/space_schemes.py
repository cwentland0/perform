import numpy as np

from perform.constants import REAL_TYPE
from perform.higher_order_funcs import calc_cell_gradients


def calc_rhs(sol_domain, solver):
	"""
	Compute rhs function
	"""

	sol_int = sol_domain.sol_int
	sol_inlet = sol_domain.sol_inlet
	sol_outlet = sol_domain.sol_outlet
	sol_prim_full = sol_domain.sol_prim_full
	sol_cons_full = sol_domain.sol_cons_full

	# compute ghost cell state (if adjacent cell is sampled)
	# TODO: update this after higher-order contribution?
	# TODO: adapt pass to calc_boundary_state() depending on space scheme
	# TODO: assign more than just one ghost cell for higher-order schemes
	if (sol_domain.direct_samp_idxs[0] == 0):
		sol_inlet.calc_boundary_state(solver.sol_time, sol_domain.space_order,
										sol_prim=sol_int.sol_prim[:, :2],
										sol_cons=sol_int.sol_cons[:, :2])
	if (sol_domain.direct_samp_idxs[-1] == (sol_domain.mesh.num_cells - 1)):
		sol_outlet.calc_boundary_state(solver.sol_time, sol_domain.space_order,
										sol_prim=sol_int.sol_prim[:, -2:],
										sol_cons=sol_int.sol_cons[:, -2:])

	sol_domain.fill_sol_full()  # fill sol_prim_full and sol_cons_full

	# first-order approx at faces
	sol_left = sol_domain.sol_left
	sol_right = sol_domain.sol_right
	sol_left.sol_prim = sol_prim_full[:, sol_domain.flux_samp_left_idxs]
	sol_left.sol_cons = sol_cons_full[:, sol_domain.flux_samp_left_idxs]
	sol_right.sol_prim = sol_prim_full[:, sol_domain.flux_samp_right_idxs]
	sol_right.sol_cons = sol_cons_full[:, sol_domain.flux_samp_right_idxs]

	# add higher-order contribution
	if (sol_domain.space_order > 1):
		sol_prim_grad = calc_cell_gradients(sol_domain)
		sol_left.sol_prim[:, sol_domain.flux_left_extract] += \
			(sol_domain.mesh.dx / 2.0) * sol_prim_grad[:, sol_domain.grad_left_extract]
		sol_right.sol_prim[:, sol_domain.flux_right_extract] -= \
			(sol_domain.mesh.dx / 2.0) * sol_prim_grad[:, sol_domain.grad_right_extract]
		sol_left.calc_state_from_prim(calc_r=True, calc_cp=True)
		sol_right.calc_state_from_prim(calc_r=True, calc_cp=True)

	# compute fluxes
	flux = sol_domain.calc_flux()

	# compute rhs
	sol_domain.sol_int.rhs[:, sol_domain.direct_samp_idxs] = \
		flux[:, sol_domain.flux_rhs_idxs] - flux[:, sol_domain.flux_rhs_idxs + 1]
	sol_int.rhs[:, sol_domain.direct_samp_idxs] /= sol_domain.mesh.dx

	# compute source term
	if solver.source_on:
		calc_source(sol_domain, solver)
		sol_int.rhs[3:, sol_domain.direct_samp_idxs] += \
			sol_int.source[:, sol_domain.direct_samp_idxs]
	
def calc_source(sol_domain, solver):
	"""
	Compute chemical source term
	"""

	# TODO: expand to multiple global reactions
	# TODO: expand to general reaction w/ reverse direction
	# TODO: check that this works for more than a two-species reaction

	gas = sol_domain.gas_model

	temp = sol_domain.sol_int.sol_prim[2, sol_domain.direct_samp_idxs]
	rho = sol_domain.sol_int.sol_cons[[0], sol_domain.direct_samp_idxs]

	# NOTE: act_energy here is -Ea/R
	# TODO: account for temperature exponential
	wf = gas.pre_exp_fact * np.exp(gas.act_energy / temp)

	rho_mass_frac = rho * sol_domain.sol_int.mass_fracs_full

	# TODO: this can be done in pre-processing
	spec_idxs = np.squeeze(np.argwhere(gas.nu_arr != 0.0))

	wf = np.product(wf[None, :] * np.power((rho_mass_frac[spec_idxs, :]
		/ gas.mol_weights[spec_idxs, None]), gas.nu_arr[spec_idxs, None]), axis=0)
	wf = np.amin(np.minimum(wf[None, :],
		rho_mass_frac[spec_idxs, :] / solver.dt), axis=0)
	sol_domain.sol_int.wf = wf

	source = sol_domain.sol_int.source
	source[gas.mass_frac_slice[:, None], sol_domain.direct_samp_idxs[None, :]] = \
		-gas.mol_weight_nu[gas.mass_frac_slice, None] * wf[None, :]

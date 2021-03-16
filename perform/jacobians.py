import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix, dia_matrix, diags

import perform.constants as const


def calc_d_sol_prim_d_sol_cons(sol_int):
	"""
	Compute the Jacobian of the conservative state w/r/t/ the primitive state
	This appears as Gamma^{-1} in the pyGEMS documentation
	"""

	# TODO: some repeated calculations
	# TODO: add option for preconditioning d_rho_d_press

	gas = sol_int.gas_model

	gamma_matrix_inv = np.zeros((gas.num_eqs, gas.num_eqs, sol_int.num_cells))

	# for clarity
	rho = sol_int.sol_cons[0, :]
	press = sol_int.sol_prim[0, :]
	vel = sol_int.sol_prim[1, :]
	temp = sol_int.sol_prim[2, :]
	mass_fracs = sol_int.sol_prim[3:, :]

	d_rho_d_press = sol_int.d_rho_d_press
	d_rho_d_temp = sol_int.d_rho_d_temp
	d_rho_d_mass_frac = sol_int.d_rho_d_mass_frac
	d_enth_d_press = sol_int.d_enth_d_press
	d_enth_d_temp = sol_int.d_enth_d_temp
	d_enth_d_mass_frac = sol_int.d_enth_d_mass_frac
	h0 = sol_int.h0

	# some reused terms
	d = (rho * d_rho_d_press * d_enth_d_temp
		- d_rho_d_temp * (rho * d_enth_d_press - 1.0))
	vel_sq = np.square(vel)

	# density row
	gamma_matrix_inv[0, 0, :] = \
		(rho * d_enth_d_temp + d_rho_d_temp * (h0 - vel_sq)
		+ np.sum(mass_fracs * (d_rho_d_mass_frac * d_enth_d_temp[None, :]
		- d_rho_d_temp[None, :] * d_enth_d_mass_frac), axis=0)) / d
	gamma_matrix_inv[0, 1, :] = vel * d_rho_d_temp / d
	gamma_matrix_inv[0, 2, :] = -d_rho_d_temp / d
	gamma_matrix_inv[0, 3:, :] = \
		(d_rho_d_temp[None, :] * d_enth_d_mass_frac
		- d_rho_d_mass_frac * d_enth_d_temp[None, :]) / d[None, :]

	# momentum row
	gamma_matrix_inv[1, 0, :] = -vel / rho
	gamma_matrix_inv[1, 1, :] = 1.0 / rho

	# energy row
	gamma_matrix_inv[2, 0, :] = \
		((-d_rho_d_press * (h0 - vel_sq) - (rho * d_enth_d_press - 1.0)
		+ np.sum(mass_fracs * ((rho * d_rho_d_press)[None, :] * d_enth_d_mass_frac
		+ d_rho_d_mass_frac * (rho * d_enth_d_press - 1.0)[None, :]), axis=0) / rho)
		/ d)
	gamma_matrix_inv[2, 1, :] = -vel * d_rho_d_press / d
	gamma_matrix_inv[2, 2, :] = d_rho_d_press / d
	gamma_matrix_inv[2, 3:, :] = \
		(-((rho * d_rho_d_press)[None, :] * d_enth_d_mass_frac
		+ d_rho_d_mass_frac * (rho * d_enth_d_press - 1.0)[None, :])
		/ (rho * d)[None, :])

	# species row(s)
	gamma_matrix_inv[3:, 0, :] = -mass_fracs / rho[None, :]
	for i in range(3, gas.num_eqs):
		gamma_matrix_inv[i, i, :] = 1.0 / rho

	return gamma_matrix_inv


def calc_d_sol_cons_d_sol_prim(sol_int):
	"""
	Compute the Jacobian of conservative state w/r/t the primitive state
	This appears as Gamma in the pyGEMS documentation
	"""

	# TODO: add option for preconditioning d_rho_d_press

	gas = sol_int.gas_model

	gamma_matrix = np.zeros((gas.num_eqs, gas.num_eqs, sol_int.num_cells))

	# for clarity
	rho = sol_int.sol_cons[0, :]
	press = sol_int.sol_prim[0, :]
	vel = sol_int.sol_prim[1, :]
	temp = sol_int.sol_prim[2, :]
	mass_fracs = sol_int.sol_prim[3:, :]

	d_rho_d_press = sol_int.d_rho_d_press
	d_rho_d_temp = sol_int.d_rho_d_temp
	d_rho_d_mass_frac = sol_int.d_rho_d_mass_frac
	d_enth_d_press = sol_int.d_enth_d_press
	d_enth_d_temp = sol_int.d_enth_d_temp
	d_enth_d_mass_frac = sol_int.d_enth_d_mass_frac
	h0 = sol_int.h0

	# density row
	gamma_matrix[0, 0, :] = d_rho_d_press
	gamma_matrix[0, 2, :] = d_rho_d_temp
	gamma_matrix[0, 3:, :] = d_rho_d_mass_frac

	# momentum row
	gamma_matrix[1, 0, :] = vel * d_rho_d_press
	gamma_matrix[1, 1, :] = rho
	gamma_matrix[1, 2, :] = vel * d_rho_d_temp
	gamma_matrix[1, 3:, :] = vel[None, :] * d_rho_d_mass_frac

	# total energy row
	gamma_matrix[2, 0, :] = d_rho_d_press * h0 + rho * d_enth_d_press - 1.0
	gamma_matrix[2, 1, :] = rho * vel
	gamma_matrix[2, 2, :] = d_rho_d_temp * h0 + rho * d_enth_d_temp
	gamma_matrix[2, 3:, :] = (h0[None, :] * d_rho_d_mass_frac
							+ rho[None, :] * d_enth_d_mass_frac)

	# species row
	gamma_matrix[3:, 0, :] = \
		mass_fracs[gas.mass_frac_slice, :] * d_rho_d_press[None, :]
	gamma_matrix[3:, 2, :] = \
		mass_fracs[gas.mass_frac_slice, :] * d_rho_d_temp[None, :]
	for i in range(3, gas.num_eqs):
		for j in range(3, gas.num_eqs):
			gamma_matrix[i, j, :] = ((i == j) * rho
									+ mass_fracs[i - 3, :] * d_rho_d_mass_frac[j - 3, :])

	return gamma_matrix


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


def calc_d_inv_flux_d_sol_prim(sol):
	"""
	Compute Jacobian of inviscid flux vector with respect to primitive state
	Here, sol should be the solutionPhys associated with the left/right face state
	"""

	gas = sol.gas_model

	d_flux_d_sol_prim = np.zeros((gas.num_eqs, gas.num_eqs, sol.num_cells))

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


def calc_d_visc_flux_d_sol_prim(sol_ave):
	"""
	Compute Jacobian of viscous flux vector with respect to the primitive state
	sol_ave is the solutionPhys associated with
	the face state used to calculate the viscous flux
	"""

	gas = sol_ave.gas_model

	d_flux_d_sol_prim = np.zeros((gas.num_eqs, gas.num_eqs,
									sol_ave.num_cells))

	# momentum equation row
	d_flux_d_sol_prim[1, 1, :] = 4.0 / 3.0 * sol_ave.dyn_visc_mix

	# energy equation row
	d_flux_d_sol_prim[2, 1, :] = \
		4.0 / 3.0 * sol_ave.sol_prim[1, :] * sol_ave.dyn_visc_mix
	d_flux_d_sol_prim[2, 2, :] = sol_ave.therm_cond_mix
	d_flux_d_sol_prim[2, 3:, :] = \
		(sol_ave.sol_cons[[0], :]
		* (sol_ave.mass_diff_mix[gas.mass_frac_slice, :]
		* sol_ave.hi[gas.mass_frac_slice, :]
		- sol_ave.mass_diff_mix[[-1], :] * sol_ave.hi[[-1], :]))

	# species transport row
	# TODO: vectorize
	for i in range(3, gas.num_eqs):
		d_flux_d_sol_prim[i, i, :] = \
			sol_ave.sol_cons[0, :] * sol_ave.mass_diff_mix[i - 3, :]

	return d_flux_d_sol_prim


def calc_d_roe_flux_d_sol_prim(sol_domain, solver):
	"""
	Compute the gradient of the inviscid and viscous fluxes
	with respect to the primitive state
	"""

	roe_diss = sol_domain.roe_diss.copy()

	d_flux_left_d_sol_prim_left = \
		calc_d_inv_flux_d_sol_prim(sol_domain.sol_left)
	d_flux_right_d_sol_prim_right = \
		calc_d_inv_flux_d_sol_prim(sol_domain.sol_right)

	if solver.visc_scheme > 0:
		d_visc_flux_d_sol_prim = \
			calc_d_visc_flux_d_sol_prim(sol_domain.sol_ave)
		d_flux_left_d_sol_prim_left -= d_visc_flux_d_sol_prim
		d_flux_right_d_sol_prim_right -= d_visc_flux_d_sol_prim

	d_flux_left_d_sol_prim_left *= (0.5 / solver.mesh.dx)
	d_flux_right_d_sol_prim_right *= (0.5 / solver.mesh.dx)

	roe_diss *= (0.5 / solver.mesh.dx)

	# Jacobian wrt current cell
	d_flux_d_sol_prim = \
		((d_flux_left_d_sol_prim_left[:, :, 1:] + roe_diss[:, :, 1:])
		+ (-d_flux_right_d_sol_prim_right[:, :, :-1] + roe_diss[:, :, :-1]))

	# Jacobian wrt left neighbor
	d_flux_d_sol_prim_left = \
		(-d_flux_left_d_sol_prim_left[:, :, 1:-1] - roe_diss[:, :, :-2])

	# Jacobian wrt right neighbor
	d_flux_d_sol_prim_right = \
		(d_flux_right_d_sol_prim_right[:, :, 1:-1] - roe_diss[:, :, 2:])

	return d_flux_d_sol_prim, d_flux_d_sol_prim_left, d_flux_d_sol_prim_right


def calc_d_res_d_sol_prim(sol_domain, solver):
	"""
	Compute Jacobian of the RHS function (i.e. fluxes and sources)
	"""

	sol_int = sol_domain.sol_int
	sol_inlet = sol_domain.sol_inlet
	sol_outlet = sol_domain.sol_outlet
	gas = sol_domain.gas_model

	# stagnation enthalpy and derivatives of density and enthalpy
	sol_int.hi = gas.calc_spec_enth(sol_int.sol_prim[2, :])
	sol_int.h0 = gas.calc_stag_enth(sol_int.sol_prim[1, :],
									sol_int.mass_fracs_full,
									spec_enth=sol_int.hi)
	sol_int.d_rho_d_press, sol_int.d_rho_d_temp, sol_int.d_rho_d_mass_frac = \
		gas.calc_dens_derivs(sol_int.sol_cons[0, :],
								wrt_press=True, pressure=sol_int.sol_prim[0, :],
								wrt_temp=True, temperature=sol_int.sol_prim[2, :],
								wrt_spec=True, mix_mol_weight=sol_int.mw_mix)

	sol_int.d_enth_d_press, sol_int.d_enth_d_temp, sol_int.d_enth_d_mass_frac = \
		gas.calc_stag_enth_derivs(wrt_press=True,
									wrt_temp=True, mass_fracs=sol_int.sol_prim[3:, :],
									wrt_spec=True, spec_enth=sol_int.hi)

	# TODO: conditional path for other flux schemes
	# *_l is contribution to lower block diagonal, *_r is to upper block diagonal
	d_flux_d_sol_prim, d_flux_d_sol_prim_left, d_flux_d_sol_prim_right = \
		calc_d_roe_flux_d_sol_prim(sol_domain, solver)
	d_rhs_d_sol_prim = d_flux_d_sol_prim.copy()

	# contribution to main block diagonal from source term Jacobian
	if solver.source_on:
		d_source_d_sol_prim = \
			calc_d_source_d_sol_prim(sol_int, sol_domain.time_integrator.dt)
		d_rhs_d_sol_prim -= d_source_d_sol_prim

	# TODO: make this specific for each implicitIntegrator
	dt_coeff_idx = min(solver.iter, sol_domain.time_integrator.time_order) - 1
	dt_inv = (sol_domain.time_integrator.coeffs[dt_coeff_idx][0]
				/ sol_domain.time_integrator.dt)

	# modifications depending on whether dual-time integration is being used
	if sol_domain.time_integrator.dual_time:

		# contribution to main block diagonal from solution Jacobian
		gamma_matrix = calc_d_sol_cons_d_sol_prim(sol_int)
		if sol_domain.time_integrator.adapt_dtau:
			dtauInv = calc_adaptive_dtau(sol_domain, gamma_matrix, solver)
		else:
			dtauInv = (1. / sol_domain.time_integrator.dtau
				* np.ones(sol_int.num_cells, dtype=const.REAL_TYPE))

		d_rhs_d_sol_prim += gamma_matrix * (dtauInv[None, None, :] + dt_inv)

		# assemble sparse Jacobian from main, upper, and lower block diagonals
		res_jacob = res_jacob_assemble(d_rhs_d_sol_prim,
										d_flux_d_sol_prim_left,
										d_flux_d_sol_prim_right,
										sol_int)

	else:
		# TODO: this is hilariously inefficient,
		# 	need to make Jacobian functions w/r/t conservative state
		# 	Convergence is also noticeably worse, since this is approximate
		# 	Transposes are due to matmul assuming
		# 	stacks are in first index, maybe a better way to do this?
		gamma_matrix_inv = \
			np.transpose(calc_d_sol_prim_d_sol_cons(sol_int),
						axes=(2, 0, 1))
		d_rhs_d_sol_cons = \
			np.transpose(np.transpose(d_rhs_d_sol_prim, axes=(2, 0, 1))
						@ gamma_matrix_inv,
						axes=(1, 2, 0))
		d_flux_d_sol_cons_left = \
			np.transpose(np.transpose(d_flux_d_sol_prim_left, axes=(2, 0, 1))
						@ gamma_matrix_inv[:-1, :, :],
						axes=(1, 2, 0))
		d_flux_d_sol_cons_right = \
			np.transpose(np.transpose(d_flux_d_sol_prim_right, axes=(2, 0, 1))
						@ gamma_matrix_inv[1:, :, :],
						axes=(1, 2, 0))

		dtMat = np.repeat(dt_inv * np.eye(gas.num_eqs)[:, :, None],
							sol_int.num_cells, axis=2)
		d_rhs_d_sol_cons += dtMat

		res_jacob = res_jacob_assemble(d_rhs_d_sol_cons,
										d_flux_d_sol_cons_left,
										d_flux_d_sol_cons_right,
										sol_int)

	return res_jacob


def calc_adaptive_dtau(sol_domain, gamma_matrix, solver):
	"""
	Adapt dtau for each cell based on user input constraints and local wave speed
	"""

	# TODO: move this to implicitIntegrator
	sol_int = sol_domain.sol_int
	gas_model = sol_domain.gas_model

	# compute initial dtau from input cfl and srf (max characteristic speed)
	# srf is computed in calcInvFlux
	dtaum = 1.0 * solver.mesh.dx / sol_domain.sol_int.srf
	dtau = sol_domain.time_integrator.cfl * dtaum

	# limit by von Neumann number
	if solver.visc_scheme > 0:
		# TODO: calculating this is stupidly expensive, figure out a workaround
		sol_int.dyn_visc_mix = \
			gas_model.calc_mix_dynamic_visc(temperature=sol_int.sol_prim[2, :],
											mass_fracs=sol_int.sol_prim[3:, :])
		nu = sol_int.dyn_visc_mix / sol_int.sol_cons[0, :]
		dtau = np.minimum(dtau,
				sol_domain.time_integrator.vnn * np.square(solver.mesh.dx) / nu)
		dtaum = np.minimum(dtaum, 3.0 / nu)

	# limit dtau
	# TODO: implement solutionChangeLimitedTimeStep from gems_precon.f90

	return 1.0 / dtau


def res_jacob_assemble(center_block, lower_block, upper_block, sol_int):
	'''
	Reassemble residual Jacobian into a sparse 2D array for linear solve
	'''

	# TODO: my God, this is still the single most expensive operation
	# 	How can this be any simpler/faster??? Preallocating "data" is *slower*

	jacob_dim = sol_int.jacob_dim

	data = np.concatenate((center_block.ravel("C"),
							lower_block.ravel("C"),
							upper_block.ravel("C")))
	res_jacob = \
		csr_matrix((data, (sol_int.jacob_row_idxs, sol_int.jacob_col_idxs)),
					shape=(jacob_dim, jacob_dim), dtype=const.REAL_TYPE)

	return res_jacob

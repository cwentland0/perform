import numpy as np

from perform.constants import REAL_TYPE
from perform.reaction.reaction import Reaction

# TODO: none of this works for multiple reactions


class FiniteRateGlobalReaction(Reaction):
	"""
	Finite rate Arrhenius reaction model assuming
	global reactions (i.e. only forwards)
	"""

	def __init__(self, gas, gas_dict):

		super().__init__()

		# Arrhenius factors
		self.nu = gas_dict["nu"].astype(REAL_TYPE)
		self.nu_arr = gas_dict["nu_arr"].astype(REAL_TYPE)
		self.act_energy = float(gas_dict["act_energy"])
		self.pre_exp_fact = float(gas_dict["pre_exp_fact"])

		# Some precomputations
		self.mol_weight_nu = gas.mol_weights * self.nu
		self.reac_idxs = np.squeeze(np.argwhere(self.nu_arr != 0.0))

	def calc_source(self, sol, dt, samp_idxs=None):
		"""
		Compute chemical source term
		"""

		# TODO: check that this works for more than a two-species reaction

		gas = sol.gas_model

		temp = sol.sol_prim[2, samp_idxs]
		rho = sol.sol_cons[[0], samp_idxs]

		# NOTE: act_energy here is -Ea/R
		# TODO: change activation energy from -Ea/R to Ea
		# TODO: account for temperature exponential
		wf = self.pre_exp_fact * np.exp(self.act_energy / temp)

		rho_mass_frac = rho * sol.mass_fracs_full[:, samp_idxs]

		wf = np.product(wf[None, :]
			* np.power((rho_mass_frac[self.reac_idxs, :]
			/ gas.mol_weights[self.reac_idxs, None]),
			self.nu_arr[self.reac_idxs, None]), axis=0)
		wf = np.amin(np.minimum(wf[None, :],
			rho_mass_frac[self.reac_idxs, :] / dt), axis=0)

		source = -self.mol_weight_nu[gas.mass_frac_slice, None] * wf[None, :]

		return source, wf

	def calc_jacob_prim(self, sol):
		"""
		Compute source term Jacobian
		"""

		# TODO: does not account for temperature exponent in Arrhenius rate
		# TODO: need to check that this works for more than a two-species reaction

		gas = sol.gas_model

		jacob = np.zeros((gas.num_eqs, gas.num_eqs, sol.num_cells))

		rho = sol.sol_cons[0, :]
		press = sol.sol_prim[0, :]
		temp = sol.sol_prim[2, :]
		mass_fracs = sol.sol_prim[3:, :]

		d_rho_d_press = sol.d_rho_d_press
		d_rho_d_temp = sol.d_rho_d_temp
		d_rho_d_mass_frac = sol.d_rho_d_mass_frac
		wf = sol.wf

		spec_idxs = np.squeeze(np.argwhere(self.nu_arr != 0.0))

		# TODO: not correct for multi-reaction
		wf_div_rho = (np.sum(wf[None, :] * self.nu_arr[spec_idxs, None], axis=0)
						/ rho[None, :])

		# subtract, as activation energy already set as negative
		d_wf_d_temp = (wf_div_rho * d_rho_d_temp[None, :]
			- wf * self.act_energy / temp**2)
		d_wf_d_press = wf_div_rho * d_rho_d_press[None, :]

		# TODO: incorrect for multi-reaction, should be [numSpec, numReac, num_cells]
		d_wf_d_mass_frac = wf_div_rho * d_rho_d_mass_frac
		for i in range(gas.num_species):
			pos_mf_idxs = np.nonzero(mass_fracs[i, :] > 0.0)[0]
			d_wf_d_mass_frac[i, pos_mf_idxs] += \
				wf[0, pos_mf_idxs] * self.nu_arr[i] / mass_fracs[i, pos_mf_idxs]

		# TODO: for multi-reaction, should be a summation over the reactions here
		jacob[3:, 0, :] = \
			-self.mol_weight_nu[gas.mass_frac_slice, None] * d_wf_d_press
		jacob[3:, 2, :] = \
			-self.mol_weight_nu[gas.mass_frac_slice, None] * d_wf_d_temp

		# TODO: this is totally wrong for multi-reaction
		for i in range(gas.num_species):
			jacob[3:, 3 + i, :] = \
				-self.mol_weight_nu[[i], None] * d_wf_d_mass_frac[[i], :]

		return jacob

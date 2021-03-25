import numpy as np

from perform.constants import REAL_TYPE
from perform.input_funcs import catch_list, catch_input
from perform.reaction.reaction import Reaction

# TODO: none of this works for multiple reactions


class FiniteRateIrrevReaction(Reaction):
    """
    Finite rate Arrhenius reaction model assuming
    irreversible reactions (i.e. only forwards)
    """

    def __init__(self, gas, gas_dict):

        super().__init__()

        self.num_reactions = catch_input(gas_dict, "num_reactions", 0)
        assert self.num_reactions > 0, "Must have num_reactions > 0 if requesting a reaction model"

        # Arrhenius factors
        self.nu = np.asarray(catch_list(gas_dict, "nu", [[-999.9]], len_highest=self.num_reactions), dtype=REAL_TYPE)
        self.nu_arr = np.asarray(
            catch_list(gas_dict, "nu_arr", [[-999.9]], len_highest=self.num_reactions), dtype=REAL_TYPE
        ).T
        self.pre_exp_fact = catch_list(gas_dict, "pre_exp_fact", [-1.0])
        self.temp_exp = catch_list(gas_dict, "temp_exp", [-1.0])
        self.act_energy = catch_list(gas_dict, "act_energy", [-1.0])

        # Some precomputations
        self.mol_weight_nu = gas.mol_weights[None, :] * self.nu
        self.spec_reac_idxs = self.nu_arr != 0.0

    def calc_source(self, sol, dt, samp_idxs=None):
        """
        Compute chemical source term
        """

        gas = sol.gas_model

        # TODO: squeeze this if samp_idxs=None?
        temp = sol.sol_prim[2, samp_idxs]
        rho = sol.sol_cons[[0], samp_idxs]
        rho_mass_frac = rho * sol.mass_fracs_full[:, samp_idxs]
        wf = [None] * self.num_reactions

        # NOTE: act_energy here is -Ea/R
        # TODO: change activation energy from -Ea/R to Ea
        for reac_idx in range(self.num_reactions):

            if self.temp_exp != 0.0:
                pre_exp = self.pre_exp_fact[reac_idx] * np.power(temp, self.temp_exp[reac_idx])
            else:
                pre_exp = self.pre_exp_fact[reac_idx]
            wf[reac_idx] = pre_exp * np.exp(self.act_energy[reac_idx] / temp)

            wf[reac_idx] = np.product(
                wf[reac_idx][None, :]
                * np.power(
                    (
                        rho_mass_frac[self.spec_reac_idxs[:, reac_idx], :]
                        / gas.mol_weights[self.spec_reac_idxs[:, reac_idx], None]
                    ),
                    np.expand_dims(self.nu_arr[self.spec_reac_idxs[:, reac_idx], reac_idx], axis=1),
                ),
                axis=0,
            )

            wf[reac_idx] = np.amin(
                np.minimum(wf[reac_idx][None, :], rho_mass_frac[self.spec_reac_idxs[:, reac_idx], :] / dt), axis=0
            )

            # TODO: gotta be a more elegant way of dealing with unexpected length due to samp_idxs
            source = -self.mol_weight_nu[reac_idx, gas.mass_frac_slice, None] * wf[reac_idx][None, :]
            if reac_idx == 0:
                source_out = source.copy()
            else:
                source_out += source

        return source_out, wf

    def calc_jacob_prim(self, sol):
        """
        Compute source term Jacobian
        """

        # TODO: does not account for temperature exponent in Arrhenius rate
        # TODO: need to check that this works for
        # more than a two-species reaction

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
        wf_div_rho = np.sum(wf[None, :] * self.nu_arr[spec_idxs, None], axis=0) / rho[None, :]

        # subtract, as activation energy already set as negative
        # TODO: reverse negation after changing Ea to positive representation, divide by R
        pre_exp = -self.act_energy / temp ** 2
        if self.temp_exp != 0.0:
            pre_exp += self.temp_exp / temp
        d_wf_d_temp = wf_div_rho * d_rho_d_temp[None, :] + wf * pre_exp

        d_wf_d_press = wf_div_rho * d_rho_d_press[None, :]

        # TODO: incorrect for multi-reaction,
        # should be [numSpec, numReac, num_cells]
        d_wf_d_mass_frac = wf_div_rho * d_rho_d_mass_frac
        for i in range(gas.num_species):
            pos_mf_idxs = np.nonzero(mass_fracs[i, :] > 0.0)[0]
            d_wf_d_mass_frac[i, pos_mf_idxs] += wf[0, pos_mf_idxs] * self.nu_arr[i] / mass_fracs[i, pos_mf_idxs]

        # TODO: for multi-reaction,
        # should be a summation over the reactions here
        jacob[3:, 0, :] = -self.mol_weight_nu[gas.mass_frac_slice, None] * d_wf_d_press
        jacob[3:, 2, :] = -self.mol_weight_nu[gas.mass_frac_slice, None] * d_wf_d_temp

        # TODO: this is totally wrong for multi-reaction
        for i in range(gas.num_species):
            jacob[3:, 3 + i, :] = -self.mol_weight_nu[[i], None] * d_wf_d_mass_frac[[i], :]

        return jacob

import numpy as np

from perform.constants import REAL_TYPE, R_UNIV
from perform.input_funcs import catch_list, catch_input
from perform.reaction_model.reaction_model import ReactionModel


class FiniteRateIrrevReaction(ReactionModel):
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
        )
        self.pre_exp_fact = np.asarray(catch_list(gas_dict, "pre_exp_fact", [-1.0]), dtype=REAL_TYPE)
        self.temp_exp = np.asarray(catch_list(gas_dict, "temp_exp", [-1.0]), dtype=REAL_TYPE)
        self.act_energy = np.asarray(catch_list(gas_dict, "act_energy", [-1.0]), dtype=REAL_TYPE)

        assert self.nu.shape == (self.num_reactions, gas.num_species_full)
        assert self.nu_arr.shape == (self.num_reactions, gas.num_species_full)

        assert self.pre_exp_fact.size == self.num_reactions
        assert self.temp_exp.size == self.num_reactions
        assert self.act_energy.size == self.num_reactions

        # Some precomputations
        self.mol_weight_nu = gas.mol_weights[None, :] * self.nu

    def calc_source(self, sol, dt, samp_idxs=None):
        """
        Compute chemical source term
        """

        # TODO: for larger mechanisms, definitely a lot of wasted flops where nu = 0.0

        gas = sol.gas_model

        # TODO: squeeze this if samp_idxs=None?
        temp = sol.sol_prim[2, samp_idxs]
        rho = sol.sol_cons[[0], samp_idxs]
        rho_mass_frac = rho * sol.mass_fracs_full[:, samp_idxs]

        # rate-of-progress
        pre_exp = self.pre_exp_fact[:, None] * np.power(temp[None, :], self.temp_exp[:, None])
        wf = pre_exp * np.exp((-self.act_energy[:, None] / R_UNIV) / temp[None, :])
        wf *= np.product(
            np.power((rho_mass_frac / gas.mol_weights[:, None])[None, :, :], self.nu_arr[:, :, None]), axis=1
        )

        # threshold
        wf = np.minimum(wf, rho_mass_frac[gas.mass_frac_slice, :] / dt)

        source = -np.sum(self.mol_weight_nu[:, gas.mass_frac_slice, None] * wf[:, None, :], axis=0)

        return source, wf

    def calc_jacob_prim(self, sol_int):
        """
        Compute source term Jacobian
        """

        gas = sol_int.gas_model

        jacob = np.zeros((gas.num_eqs, gas.num_eqs, sol_int.num_cells))

        rho = sol_int.sol_cons[0, :]
        press = sol_int.sol_prim[0, :]
        temp = sol_int.sol_prim[2, :]
        mass_fracs = sol_int.sol_prim[3:, :]

        # assumes density derivatives have been precomputed
        # TODO: make sure this happens always, not just in Roe flux
        d_rho_d_press = sol_int.d_rho_d_press
        d_rho_d_temp = sol_int.d_rho_d_temp
        d_rho_d_mass_frac = sol_int.d_rho_d_mass_frac
        wf = sol_int.wf

        wf_div_rho = np.sum(wf[:, None, :] * self.nu_arr[:, :, None], axis=1) / rho[None, :]

        d_wf_d_press = wf_div_rho * d_rho_d_press

        # subtract, as activation energy already set as negative
        # TODO: reverse negation after changing Ea to positive representation, divide by R
        pre_exp = self.temp_exp[:, None] / temp[None, :] + (self.act_energy[:, None] / R_UNIV) / temp[None, :] ** 2
        d_wf_d_temp = pre_exp * wf + wf_div_rho * d_rho_d_temp[None, :]

        d_wf_d_mass_fracs = wf_div_rho[:, None, :] * d_rho_d_mass_frac[None, :, :]
        for spec_idx in range(gas.num_species):
            pos_mf_idxs = np.nonzero(mass_fracs[spec_idx, :] > 0.0)[0]
            if pos_mf_idxs.size == 0:
                continue
            d_wf_d_mass_fracs[:, spec_idx, pos_mf_idxs] += (
                wf[:, pos_mf_idxs] * self.nu_arr[:, spec_idx, None] / mass_fracs[None, spec_idx, pos_mf_idxs]
            )

        # compute Jacobian
        jacob[3:, 0, :] = -np.sum(self.mol_weight_nu[:, gas.mass_frac_slice, None] * d_wf_d_press[:, None, :], axis=0)
        jacob[3:, 2, :] = -np.sum(self.mol_weight_nu[:, gas.mass_frac_slice, None] * d_wf_d_temp[:, None, :], axis=0)

        # TODO: I'm a little unsure this is correct
        jacob[3:, 3:, :] = -np.sum(
            self.mol_weight_nu[:, gas.mass_frac_slice, None, None] * d_wf_d_mass_fracs[:, None, :, :], axis=0
        )

        return jacob

from perform.rom.rom_variable_mapping.rom_variable_mapping import RomVariableMapping


class PrimVariableMapping(RomVariableMapping):
    """Trivial mapping to primitive state.

    RomDomains with this mapping are assumed to map to the primitive state,
    given by [pressure, velocity, temperature, species mass fraction].
    """

    def __init__(self, sol_domain, rom_domain):

        self.num_vars = sol_domain.gas_model.num_eqs
        self.is_complete = True

        super().__init__(rom_domain)

    def get_variables_from_state(self, sol_domain):
        """Retrieves mapped state from current SolutionDomain state."""

        return sol_domain.sol_int.sol_prim.copy()

    def get_variable_hist_from_state_hist(self, sol_domain):
        """Retrieves mapped state from SolutionDomain state history."""

        return sol_domain.sol_int.sol_hist_prim.copy()

    def update_full_state(self, sol_domain, rom_domain):
        """Collects model internal states and updates full SolutionDomain state, including thermo/transport.

        PrimVariableMapping trivially collects and updates the solution from the primitive state.
        No additional transformations are required.
        """

        for rom_model in rom_domain.model_list:
            var_idxs = rom_model.var_idxs
            sol_domain.sol_int.sol_prim[var_idxs, :] = rom_model.sol.copy()

        sol_domain.sol_int.update_state(from_prim=True)

        # map back to account for any mass fraction thresholding
        for rom_model in rom_domain.model_list:
            var_idxs = rom_model.var_idxs
            rom_model.sol[:, :] = sol_domain.sol_int.sol_prim[var_idxs, :].copy()

    def update_state_hist(self, sol_domain, rom_domain):
        """Collects model internal state history and updates relevant state history"""

        sol_int = sol_domain.sol_int

        # loop over history
        for sol_idx in range(len(sol_domain.sol_int.sol_hist_prim)):

            # loop over models, collect state
            for rom_model in rom_domain.model_list:
                var_idxs = rom_model.var_idxs
                sol_int.sol_hist_prim[sol_idx][var_idxs, :] = rom_model.sol_hist[sol_idx].copy()

            # threshold mass fractions
            mass_fracs = sol_int.gas_model.get_mass_frac_array(sol_prim_in=sol_int.sol_hist_prim[sol_idx])
            mass_fracs = sol_int.gas_model.calc_all_mass_fracs(mass_fracs, threshold=True)
            sol_int.sol_hist_prim[sol_idx][3:, :] = mass_fracs[:-1, :]

            # update conservative state history
            sol_int.sol_hist_cons[sol_idx] = sol_int.calc_cons_from_prim(sol_int.sol_hist_prim[sol_idx])

            # map back to account for any mass fraction thresholding
            for rom_model in rom_domain.model_list:
                var_idxs = rom_model.var_idxs
                rom_model.sol_hist[sol_idx][:, :] = sol_int.sol_hist_prim[sol_idx][var_idxs, :].copy()

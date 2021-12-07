from perform.solution.solution_phys import SolutionPhys
from perform.rom.rom_variable_mapping.rom_variable_mapping import RomVariableMapping


class ConsVariableMapping(RomVariableMapping):
    """Trivial mapping to conservative state.

    RomDomains with this mapping are assumed to map to the conservative state,
    given by [density, momentum, energy, density-weighted species mass fraction].
    """

    def __init__(self, sol_domain, rom_domain):

        self.num_vars = sol_domain.gas_model.num_eqs
        self.is_complete = True

        super().__init__(rom_domain)

    def get_variables_from_state(self, sol_domain):
        """Retrieves mapped state from current SolutionDomain state."""

        return sol_domain.sol_int.sol_cons.copy()

    def get_variable_hist_from_state_hist(self, sol_domain):
        """Retrieves mapped state from SolutionDomain state history."""

        return sol_domain.sol_int.sol_hist_cons.copy()

    def update_full_state(self, sol_domain, rom_domain):
        """Collects model internal states and updates full SolutionDomain state, including thermo/transport.

        ConsVariableMapping trivially collects and updates the solution from the conservative state.
        No additional transformations are required.
        """

        for rom_model in rom_domain.model_list:
            var_idxs = rom_model.var_idxs
            sol_domain.sol_int.sol_cons[var_idxs, :] = rom_model.sol.copy()

        sol_domain.sol_int.update_state(from_prim=False)

        # map back to account for any mass fraction thresholding
        for rom_model in rom_domain.model_list:
            var_idxs = rom_model.var_idxs
            rom_model.sol[:, :] = sol_domain.sol_int.sol_cons[var_idxs, :].copy()

    def update_state_hist(self, sol_domain, rom_domain):
        """Collects model internal state history and updates relevant state history"""

        sol_int = sol_domain.sol_int

        # temporary SolutionPhys to access update_state()
        sol_temp = SolutionPhys(sol_int.gas_model, sol_int.num_cells, sol_prim_in=sol_int.sol_prim, time_order=1)

        # loop over history
        for sol_idx in range(len(sol_domain.sol_int.sol_hist_cons)):

            # loop over models, collect state
            for rom_model in rom_domain.model_list:
                var_idxs = rom_model.var_idxs
                sol_int.sol_hist_cons[sol_idx][var_idxs, :] = rom_model.sol_hist[sol_idx].copy()

            # update state
            sol_temp.sol_cons = sol_int.sol_hist_cons[sol_idx].copy()
            sol_temp.update_state(from_prim=False)
            sol_int.sol_hist_cons[sol_idx][:, :] = sol_temp.sol_cons.copy()
            sol_int.sol_hist_prim[sol_idx][:, :] = sol_temp.sol_prim.copy()

            # map back to account for any changes in state update
            for rom_model in rom_domain.model_list:
                var_idxs = rom_model.var_idxs
                rom_model.sol_hist[sol_idx][:, :] = sol_int.sol_hist_cons[sol_idx][var_idxs, :].copy()


from perform.rom.rom_variable_mapping.rom_variable_mapping import RomVariableMapping

class PrimVariableMapping(RomVariableMapping):
    """Trivial mapping to primitive state.
    
    RomDomains with this mapping are assumed to map to the primitive state,
    given by [pressure, velocity, temperature, species mass fraction].
    """

    def __init__(self, sol_domain, rom_domain):

        self.num_vars = sol_domain.gas_model.num_eqs

        super().__init__(rom_domain)
        
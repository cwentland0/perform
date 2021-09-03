
from perform.rom.rom_variable_mapping.rom_variable_mapping import RomVariableMapping

class LiftedXiCVariableMapping(RomVariableMapping):
    """Mapping to lifted primitive state with specific volume and molar concentrations.
    
    RomDomains with this mapping are assumed to map to a lifted state
    given by [pressure, velocity, temperature, specific volume, species molar concentrations].
    """

    def __init__(self, sol_domain, rom_domain):

        self.num_vars = sol_domain.gas_model.num_eqs + 1  # added specific volume

        super().__init__(rom_domain)
# ROM methods
from perform.rom.rom_method.projection_method.galerkin_projection import GalerkinProjection
from perform.rom.rom_method.projection_method.lspg_projection import LSPGProjection
from perform.rom.rom_method.projection_method.mplsvt_projection import MPLSVTProjection

# variable mappings
from perform.rom.rom_variable_mapping.prim_variable_mapping import PrimVariableMapping
from perform.rom.rom_variable_mapping.cons_variable_mapping import ConsVariableMapping

# time steppers
from perform.rom.rom_time_stepper.numerical_stepper import NumericalStepper

# space mappings
from perform.rom.rom_space_mapping.linear_space_mapping import LinearSpaceMapping


def get_rom_method(rom_method, sol_domain, rom_domain):

    if rom_method == "galerkin":
        return GalerkinProjection(sol_domain, rom_domain)
    elif rom_method == "lspg":
        return LSPGProjection(sol_domain, rom_domain)
    elif rom_method == "mplsvt":
        return MPLSVTProjection(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM rom_method: " + str(rom_method))


def get_variable_mapping(var_mapping, sol_domain, rom_domain):

    if var_mapping == "primitive":
        return PrimVariableMapping(sol_domain, rom_domain)
    elif var_mapping == "conservative":
        return ConsVariableMapping(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM var_mapping: " + str(var_mapping))


def get_time_stepper(time_stepper, sol_domain, rom_domain):

    if time_stepper == "numerical":
        return NumericalStepper(sol_domain, rom_domain)
    else:
        raise ValueError("Something went wrong, rom_method set invalid time_stepper: " + str(time_stepper))


def get_space_mapping(space_mapping, sol_domain, rom_domain):

    if space_mapping == "linear":
        return LinearSpaceMapping(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM space_mapping: " + str(space_mapping))

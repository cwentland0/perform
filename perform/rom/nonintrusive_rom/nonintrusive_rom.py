from perform.rom.rom_model import RomModel

class NonIntrusiveRom(RomModel):
    """Base class for non-intrusive ROMs.

    This class provides basic functionality for non-intrusive ROMs, i.e. those that don't require access to
    the non-linear RHS term of the discretized governing equations.
    All these methods should require access to is, at most, the initial conditions.
    Whether they require a numerical time integrator is a separate matter.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)

        # If no numerical time integration, only use one step
        if not rom_domain.has_time_integrator:
            rom_domain.time_integrator.subiter_max = 1
    
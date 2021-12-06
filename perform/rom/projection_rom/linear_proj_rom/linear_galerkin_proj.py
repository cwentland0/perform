from perform.rom.projection_rom.linear_proj_rom.linear_proj_rom import LinearProjROM


class LinearGalerkinProj(LinearProjROM):
    """Class for projection-based ROM with linear decoder and Galerkin projection.

    Inherits from LinearProjROM.

    Trial basis is assumed to represent the conservative variables. Allows implicit and explicit time integration.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        trial_basis_scaled: 2D NumPy array of trial basis scaled by norm_fac_prof_cons. Precomputed for cost savings.
        hyper_reduc_operator: 2D NumPy array of gappy POD projection operator. Precomputed for cost savings.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if (rom_domain.time_integrator.time_type == "implicit") and (rom_domain.time_integrator.dual_time):
            raise ValueError("Galerkin is intended for conservative variable evolution, please set dual_time = False")

        super().__init__(model_idx, rom_domain, sol_domain)

    def calc_projector(self, sol_domain):
        """Compute RHS projection operator.

        Called by ProjectionROM.calc_rhs_low_dim() to compute projection operator which is applied to RHS function
        for explicit time integration.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
        """

        if self.hyper_reduc:
            self.projector = self.hyper_reduc_operator
        else:
            self.projector = self.trial_basis.T

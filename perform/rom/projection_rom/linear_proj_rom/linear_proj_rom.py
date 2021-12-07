from perform.rom.projection_rom.projection_rom import ProjectionROM


class LinearProjROM(ProjectionROM):
    """Base class for all linear subspace projection-based ROMs.

    Inherits from ProjectionROM. Assumes that solution decoding is computed by

    sol = cent_prof + norm_sub_prof + norm_fac_prof * (trial_basis @ code)

    Child classes must implement a calc_projector() member function if it permits explicit time integration,
    and/or a calc_d_code() member function if it permits implicit time integration.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        trial_basis:
            2D NumPy array containing latent_dim trial basis modes. Modes are flattened in C order,
            i.e. iterating first over cells, then over variables.
        hyper_reduc_dim: Number of modes to retain in hyper_reduc_basis.
        hyper_reduc_basis:
            2D NumPy array containing hyper_reduc_dim hyper-reduction basis modes. Modes are flattened in C order,
            i.e. iterating first over cells, then over variables.
        direct_samp_idxs_flat:
            NumPy array of slicing indices for slicing directly-sampled cells from
            solution-related vectors flattened in C order.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)

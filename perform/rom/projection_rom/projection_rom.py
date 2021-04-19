import numpy as np

from perform.rom.rom_model import RomModel
from perform.constants import REAL_TYPE


class ProjectionROM(RomModel):
    """Base class for projection-based reduced-order models.

    Inherits from RomModel. This class makes no assumption on the form of the decoder,
    but assumes a linear projection onto the low-dimensional space. 

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
    """

    def __init__(self, modelIdx, rom_domain, sol_domain):

        self.hyper_reduc = rom_domain.hyper_reduc

        super().__init__(modelIdx, rom_domain, sol_domain)

    def project_to_low_dim(self, projector, full_dim_arr, transpose=False):
        """Project given full-dimensional vector onto low-dimensional space via given projector.

        Assumes that full_dim_arr is either 1D array or is in [num_vars, num_cells] order.
        Further assumes that projector is already in [latent_dim, num_vars x num_cells] order.

        Args:
            projector: 2D NumPy array containing linear projector. 
            full_dim_arr: NumPy array of full-dimensional vector to be projected.
            transpose: If True, transposes projector before projecting full_dim_arr.

        Returns:
            NumPy array of low-dimensional projection of full_dim_arr.
        """

        if full_dim_arr.ndim == 2:
            full_dim_vec = full_dim_arr.flatten(order="C")
        elif full_dim_arr.ndim == 1:
            full_dim_vec = full_dim_arr.copy()
        else:
            raise ValueError("full_dim_arr must be one- or two-dimensional")

        if transpose:
            code_out = projector.T @ full_dim_vec
        else:
            code_out = projector @ full_dim_vec

        return code_out

    def calc_rhs_low_dim(self, rom_domain, sol_domain):
        """Project RHS onto low-dimensional space for explicit time integrators.

        This is a helper function called from RomDomain.advance_subiter() for explicit time integration.
        Child classes which enable explicit time integration must implement a calc_projector() member function
        to compute the projector attribute.

        Args:
            rom_domain: RomDomain within which this RomModel is contained.
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
        """

        # scale RHS
        norm_sub_prof = np.zeros(self.norm_fac_prof_cons.shape, dtype=REAL_TYPE)

        rhs_scaled = self.scale_profile(
            sol_domain.sol_int.rhs[self.var_idxs[:, None], sol_domain.direct_samp_idxs[None, :]],
            normalize=True,
            norm_fac_prof=self.norm_fac_prof_cons[:, sol_domain.direct_samp_idxs],
            norm_sub_prof=norm_sub_prof[:, sol_domain.direct_samp_idxs],
            center=False,
            inverse=False,
        )

        # calc projection operator and project
        self.calc_projector(sol_domain)
        self.rhs_low_dim = self.project_to_low_dim(self.projector, rhs_scaled, transpose=False)

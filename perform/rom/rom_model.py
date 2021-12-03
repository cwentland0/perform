import numpy as np

from perform.constants import REAL_TYPE
from perform.rom.rom_space_mapping.linear_space_mapping import LinearSpaceMapping
from perform.rom.rom_space_mapping.autoencoder_space_mapping import AutoencoderSpaceMapping
from perform.rom.rom_variable_mapping.cons_variable_mapping import ConsVariableMapping


class RomModel:
    """Base class for all ROM models.

    The RomModel class provides basic functionality that every ROM model should be equipped with,
    most importantly feature scaling and mapping from the low-dimensional state to
    the high-dimensional state ("decoding"). Feature scaling is general, but decoding operations
    are specific to child classes. Thus, RomModel only provides the high-level utility decode_sol(),
    which calls child class implementations of apply_decoder().

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        latent_dim: Dimension of low-dimensional code associated with this model.
        var_idxs: NumPy array of zero-indexed indices of state variables which this model maps to.
        num_vars: Number of state variables this model maps to.
        num_cells: Number of finite volume cells in associated SolutionDomain.
        sol_shape: Tuple shape (num_vars, num_cells) of full-dimensional state which this model maps to.
        target_cons: Boolean flag indicating whether this model maps to the conservative variables.
        code: NumPy array of unsteady low-dimensional state associated with this model.
        res:
            NumPy array of low-dimensional linear solve residual when integrating ROM ODE in time
            with an implicit time integrator.
        norm_sub_prof_cons:
            NumPy array of conservative variable subtractive normalization profile,
            if required by ROM method.
        norm_fac_prof_cons:
            NumPy array of conservative variable divisive normalization profile, if required by ROM method.
        cent_prof_cons: NumPy array of conservative variable centering profile, if required by ROM method.
        norm_sub_prof_prim:
            NumPy array of primitive variable subtractive normalization profile, if required by ROM method.
        norm_fac_prof_prim:
        NumPy array of primitive variable divisive normalization profile, if required by ROM method.
        cent_prof_prim: NumPy array of primitive variable centering profile, if required by ROM method.
    """

    def __init__(self, model_idx, sol_domain, rom_domain):

        self.model_idx = model_idx
        self.latent_dim = rom_domain.latent_dims[self.model_idx]
        self.var_idxs = np.array(rom_domain.model_var_idxs[self.model_idx], dtype=np.int32)
        self.num_vars = len(self.var_idxs)
        self.num_cells = sol_domain.mesh.num_cells
        self.sol_shape = (self.num_vars, self.num_cells)

        # initialize space mapping
        space_mapping = rom_domain.rom_dict["space_mapping"]
        if space_mapping == "linear":
            self.space_mapping = LinearSpaceMapping(sol_domain, rom_domain, self)
        elif space_mapping == "autoencoder":
            self.space_mapping = AutoencoderSpaceMapping(sol_domain, rom_domain, self)
        else:
            raise ValueError("Invalid space_mapping: " + str(space_mapping))

        # internal state and related quantities
        self.sol = np.zeros(self.sol_shape, dtype=REAL_TYPE)
        self.code = np.zeros(self.latent_dim, dtype=REAL_TYPE)
        self.d_code = np.zeros(self.latent_dim, dtype=REAL_TYPE)  # TODO: is this used?

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
        norm_sub_prof = np.zeros(self.space_mapping.norm_sub_prof.shape, dtype=REAL_TYPE)

        # if variable mapping is not conservative, need to supply conservative variable scaling
        if isinstance(rom_domain.var_mapping, ConsVariableMapping):
            norm_fac_prof = self.space_mapping.norm_fac_prof
        else:
            norm_fac_prof = self.space_mapping.norm_fac_prof_cons

        rhs_scaled = self.space_mapping.scale_profile(
            sol_domain.sol_int.rhs[self.var_idxs[:, None], sol_domain.direct_samp_idxs[None, :]],
            normalize=True,
            norm_fac_prof=norm_fac_prof[:, sol_domain.direct_samp_idxs],
            norm_sub_prof=norm_sub_prof[:, sol_domain.direct_samp_idxs],
            center=False,
            inverse=False,
        )

        # calc projection operator and project
        projector = rom_domain.rom_method.calc_projector(sol_domain, self)
        self.rhs_low_dim = rom_domain.rom_method.project_to_low_dim(projector, rhs_scaled, transpose=False)

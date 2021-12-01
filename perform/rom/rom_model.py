import numpy as np

from perform.constants import REAL_TYPE
from perform.rom.rom_space_mapping.linear_space_mapping import LinearSpaceMapping


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
        else:
            raise ValueError("Invalid space_mapping: " + str(space_mapping))

        # internal state and related quantities
        self.sol = np.zeros(self.sol_shape, dtype=REAL_TYPE)
        self.code = np.zeros(self.latent_dim, dtype=REAL_TYPE)
        self.d_code = np.zeros(self.latent_dim, dtype=REAL_TYPE)  # TODO: is this used?

    def update_sol(self, sol_domain):
        """Update solution after low-dimensional code has been updated.

        Helper function that updates full-dimensional state within the given SolutionDomain.
        This function is called within RomDomain.advance_subiter() for those ROM methods with a time integrator,
        and within RomDomain.advance_iter() for non-intrusive methods without a time integrator.

        Args:
            sol_domain: SolutionDomain with which this RomModel's containing RomDomain is associated.
        """

        if self.target_cons:
            sol_domain.sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)
        else:
            sol_domain.sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)

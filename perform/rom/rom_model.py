import os
import time

import numpy as np
from scipy.sparse import vstack

from perform.constants import REAL_TYPE


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

    def __init__(self, model_idx, rom_domain, sol_domain):

        self.model_idx = model_idx
        self.latent_dim = rom_domain.latent_dims[self.model_idx]
        self.var_idxs = np.array(rom_domain.model_var_idxs[self.model_idx], dtype=np.int32)
        self.num_vars = len(self.var_idxs)
        self.num_cells = sol_domain.mesh.num_cells
        self.sol_shape = (self.num_vars, self.num_cells)

        # Just copy some stuff for less clutter
        model_dir = rom_domain.model_dir
        self.target_cons = rom_domain.target_cons
        self.target_prim = rom_domain.target_prim

        self.code = np.zeros(self.latent_dim, dtype=REAL_TYPE)
        self.d_code = np.zeros(self.latent_dim, dtype=REAL_TYPE)

        # For implicit solve
        if rom_domain.has_time_integrator:
            if rom_domain.time_integrator.time_type == "implicit":
                self.res = np.zeros(self.latent_dim, dtype=REAL_TYPE)

        # Get normalization profiles, if necessary
        self.norm_sub_prof_cons = None
        self.norm_sub_prof_prim = None
        self.norm_fac_prof_cons = None
        self.norm_fac_prof_prim = None
        self.cent_prof_cons = None
        self.cent_prof_prim = None
        if rom_domain.has_cons_norm:
            self.norm_sub_prof_cons = self.load_feature_scaling(
                os.path.join(model_dir, rom_domain.norm_sub_cons_in[self.model_idx]), default="zeros"
            )

            self.norm_fac_prof_cons = self.load_feature_scaling(
                os.path.join(model_dir, rom_domain.norm_fac_cons_in[self.model_idx]), default="ones"
            )

        if rom_domain.has_prim_norm:
            self.norm_sub_prof_prim = self.load_feature_scaling(
                os.path.join(model_dir, rom_domain.norm_sub_prim_in[self.model_idx]), default="zeros"
            )

            self.norm_fac_prof_prim = self.load_feature_scaling(
                os.path.join(model_dir, rom_domain.norm_fac_prim_in[self.model_idx]), default="ones"
            )

        # Get centering profiles, if necessary
        # If cent_ic, just use given initial conditions
        if rom_domain.has_cons_cent:
            if rom_domain.cent_ic:
                self.cent_prof_cons = sol_domain.sol_int.sol_cons[self.var_idxs, :].copy()

            else:
                self.cent_prof_cons = self.load_feature_scaling(
                    os.path.join(model_dir, rom_domain.cent_cons_in[self.model_idx]), default="zeros"
                )

        if rom_domain.has_prim_cent:
            if rom_domain.cent_ic:
                self.cent_prof_prim = sol_domain.sol_int.sol_prim[self.var_idxs, :].copy()
            else:
                self.cent_prof_prim = self.load_feature_scaling(
                    os.path.join(model_dir, rom_domain.cent_prim_in[self.model_idx]), default="zeros"
                )

    def load_feature_scaling(self, scaling_input, default="zeros"):
        """Load a normalization or centering profile from NumPy binary.

        Args:
            scaling_input: String path to scaling profile NumPy binary.
            default: String indicating default profile if loading fails due to size mismatch or load failure.

        Returns:
            scaling_prof: NumPy array of scaling profile loaded (or default, if load failed).
        """

        try:
            # Load single complete standardization profile from file
            scaling_prof = np.load(scaling_input)
            assert scaling_prof.shape == self.sol_shape
            return scaling_prof

        except AssertionError:
            print("Standardization profile at " + scaling_input + " did not match solution shape")

        if default == "zeros":
            print("WARNING: standardization load failed or not specified, defaulting to zeros")
            time.sleep(1.0)
            scaling_prof = np.zeros(self.sol_shape, dtype=REAL_TYPE)
        elif default == "ones":
            print("WARNING: standardization load failed or not specified, defaulting to ones")
            time.sleep(1.0)
            scaling_prof = np.zeros(self.sol_shape, dtype=REAL_TYPE)
        else:
            raise ValueError("Invalid default: " + str(default))

        return scaling_prof

    def scale_profile(
        self, arr_in, normalize=True, norm_fac_prof=None, norm_sub_prof=None, center=True, cent_prof=None, inverse=False
    ):
        """(De-)centers and/or (de-)normalizes solution profile.

        Depending on argument flags, centers and/or normalizes solution profile, or de-normalizes
        and/or de-centers solution profile.

        If inverse is False:
            arr = (arr_in - cent_prof - norm_sub_prof) / norm_fac_prof

        If inverse is True:
            arr = arr_in * norm_fac_prof + norm_sub_prof + cent_prof

        Args:
            arr_in: NumPy array of solution profile to be scaled.
            normalize: Boolean flag indicating whether arr_in should be (de-)normalized.
            norm_fac_prof: NumPy array of divisive normalization profile.
            norm_sub_prof: NumPy array of subtractive normalization profile.
            center: Boolean flag indicating whether arr_in should be (de-)centered.
            cent_prof: NumPy array of centering profile.
            inverse: If True, de-normalize and de-center. If False, center and normalize.

        Returns:
            (De)-centered and/or (de)-normalized copy of arr_in.
        """

        arr = arr_in.copy()

        assert normalize or center, "Must either (de-)center or (de-)normalize."

        if normalize:
            assert norm_fac_prof is not None, "Must provide normalization division factor to normalize"
            assert norm_sub_prof is not None, "Must provide normalization subtractive factor to normalize"
        if center:
            assert cent_prof is not None, "Must provide centering profile to center"

        # de-normalize and de-center
        if inverse:
            if normalize:
                arr = self.normalize(arr, norm_fac_prof, norm_sub_prof, denormalize=True)
            if center:
                arr = self.center(arr, cent_prof, decenter=True)

        # center and normalize
        else:
            if center:
                arr = self.center(arr, cent_prof, decenter=False)
            if normalize:
                arr = self.normalize(arr, norm_fac_prof, norm_sub_prof, denormalize=False)

        return arr

    def center(self, arr_in, cent_prof, decenter=False):
        """(De)center input vector according to provided centering profile.

        Args:
            arr_in: NumPy array to be (de-)centered.
            cent_prof: NumPy array of centering profile.
            decenter: If True, decenter profile. If False, center profile.

        Returns:
            (De-)centered copy of arr_in.
        """

        if decenter:
            arr = arr_in + cent_prof
        else:
            arr = arr_in - cent_prof
        return arr

    def normalize(self, arr_in, norm_fac_prof, norm_sub_prof, denormalize=False):
        """(De)normalize input vector according to subtractive and divisive normalization profiles.

        Args:
            arr_in: NumPy array to be (de-)normalized.
            norm_fac_prof: NumPy array of divisive normalization profile.
            norm_sub_prof: NumPy array of subtractive normalization profile.
            denormalize: If True, denormalize profile. If False, normalize profile.

        Returns:
            (De-)normalized copy of arr_in.
        """

        if denormalize:
            arr = arr_in * norm_fac_prof + norm_sub_prof
        else:
            arr = (arr_in - norm_sub_prof) / norm_fac_prof
        return arr

    def decode_sol(self, code_in):
        """Compute full decoding of solution, including de-centering and de-normalization.

        Maps low-dimensional code to full-dimensional state, and de-centers and de-normalizes.
        Note that the apply_decoder is implemented within child classes, as these are specific to ROM method.

        Args:
            code_in: low-dimensional code to be decoded.

        Returns:
            Full-dimensional solution NumPy array resulting from decoding and de-scaling.
        """

        sol = self.apply_decoder(code_in)

        if self.target_cons:
            sol = self.scale_profile(
                sol,
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_cons,
                norm_sub_prof=self.norm_sub_prof_cons,
                center=True,
                cent_prof=self.cent_prof_cons,
                inverse=True,
            )

        else:
            sol = self.scale_profile(
                sol,
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_prim,
                norm_sub_prof=self.norm_sub_prof_prim,
                center=True,
                cent_prof=self.cent_prof_prim,
                inverse=True,
            )

        return sol

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

    def calc_code_norms(self):
        """Compute L1 and L2 norms of low-dimensional state linear solve residuals

        This function is called within RomDomain.calc_code_res_norms(), after which the residual norms are averaged
        across all models for an aggregate measure. Note that this measure is scaled by number of elements,
        so "L2 norm" here is really RMS.

        Returns:
            The L2 and L1 norms of the low-dimensional linear solve residual.
        """

        res_abs = np.abs(self.res)

        # L2 norm
        res_norm_l2 = np.sum(np.square(res_abs))
        res_norm_l2 /= self.latent_dim
        res_norm_l2 = np.sqrt(res_norm_l2)

        # L1 norm
        res_norm_l1 = np.sum(res_abs)
        res_norm_l1 /= self.latent_dim

        return res_norm_l2, res_norm_l1

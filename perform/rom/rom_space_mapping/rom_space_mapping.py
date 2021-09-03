import time
import os

import numpy as np

from perform.constants import REAL_TYPE

class RomSpaceMapping():
    """Base class for mapping to/from the state/latent space."""

    def __init__(self, sol_domain, rom_domain, rom_model):
        
        rom_dict = rom_domain.rom_dict
        model_idx = rom_model.model_idx
        self.latent_dim = rom_model.latent_dim
        self.sol_shape = rom_model.sol_shape

        # all mappings require scaling by default, specific methods may include additional scalings
        model_dir = rom_dict["model_dir"]
        self.cent_prof = self.load_feature_scaling(os.path.join(model_dir, rom_dict["cent_prof"][model_idx]), default="zeros")
        self.norm_fac_prof = self.load_feature_scaling(os.path.join(model_dir, rom_dict["norm_fac_prof"][model_idx]), default="ones")
        self.norm_sub_prof = self.load_feature_scaling(os.path.join(model_dir, rom_dict["norm_sub_prof"][model_idx]), default="zeros")
        if callable(getattr(rom_domain.rom_method, "load_extra_scalings", None)):
            rom_domain.rom_method.load_extra_scalings(model_idx, sol_domain, rom_domain)

        # specific mapping loading functions implemented by child classes
        self.load_mapping()

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

    def encode_decode_series(self, sol_series_in):
        """Compute encoding and decoding of a list of solution arrays"""

        if isinstance(sol_series_in, np.ndarray):
            sol_series_in = [sol_series_in]

        code_series_out = []
        sol_series_out = []
        for sol in sol_series_in:
            code_series_out.append(self.encode_sol(sol))
            sol_series_out.append(self.decode_sol(code_series_out[-1]))

        return code_series_out, sol_series_out

    def encode_sol(self, sol_in):

        sol = self.scale_profile(
            sol_in,
            normalize=True,
            norm_fac_prof=self.norm_fac_prof,
            norm_sub_prof=self.norm_sub_prof,
            center=True,
            cent_prof=self.cent_prof,
            inverse=False,
        )
        code = self.apply_encoder(sol)

        return code

    def decode_sol(self, code_in):
        """Compute full decoding of solution, including de-centering and de-normalization.

        Maps low-dimensional code to full-dimensional state, and de-centers and de-normalizes.
        Note that the apply_decoder is implemented within child classes, as these are specific to a given mapping.

        Args:
            code_in: low-dimensional code to be decoded.

        Returns:
            Full-dimensional solution NumPy array resulting from decoding and de-scaling.
        """

        sol = self.apply_decoder(code_in)
        sol = self.scale_profile(
            sol,
            normalize=True,
            norm_fac_prof=self.norm_fac_prof,
            norm_sub_prof=self.norm_sub_prof,
            center=True,
            cent_prof=self.cent_prof,
            inverse=True,
        )

        return sol

    
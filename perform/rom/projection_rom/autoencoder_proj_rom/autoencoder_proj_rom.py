import os

from perform.constants import FD_STEP_DEFAULT
from perform.rom.projection_rom.projection_rom import ProjectionROM
from perform.input_funcs import catch_input


class AutoencoderProjROM(ProjectionROM):
    """Base class for all non-linear manifold projection-based ROMs using autoencoders.
    
    Inherits from ProjectionROM. Assumes that solution decoding is computed by

    sol = cent_prof + norm_sub_prof + norm_fac_prof * decoder(code)
    
    Child classes must implement the following member functions with
    library-specific (e.g. TensorFlow-Keras, PyTorch) implementations:
    * load_model_obj()
    * check_model()
    * apply_decoder()
    * apply_encoder()
    * calc_analytical_model_jacobian()
    * calc_numerical_model_jacobian()
    * calc_model_jacobian()
    Check autoencoder_tfkeras.py for further details on these methods.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        decoder:
            Decoder model object (e.g. tf.keras.Model) which maps from the low-dimensional state
            to the full-dimensional state.
        decoder_io_dtypes: List containing the data type of the decoder input and output.
        encoder_jacob:
            Boolean flag indicating whether the method should use an encoder Jacobian approximation, if applicable.
        encoder:
            Encoder model object (e.g. tf.keras.Model) which maps from the full-dimensional state
            to the low-dimensional state. Only required if initializing the simulation from a full-dimensional
            initial conditions or the ROM method requires an encoder.
        encoder_io_dtypes: List containing the data type of the encoder input and output.
        numerical_jacob:
            Boolean flag indicating whether the decoder/encoder Jacobian should be computed numerically.
            If False (default), computes the Jacobian analytically using automatic differentiation.
        fd_step: Finite-difference step size for computing the numerical decoder/encoder Jacobian, if requested.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)

        rom_dict = rom_domain.rom_dict

        # Load decoder
        decoder_path = os.path.join(rom_domain.model_dir, rom_domain.model_files[model_idx])
        assert os.path.isfile(decoder_path), "Invalid decoder file path"
        self.decoder = self.load_model_obj(decoder_path)
        self.decoder_io_dtypes = self.check_model(decoder=True)

        # If required, load encoder
        # Encoder is required for encoder Jacobian or initializing from projection of full ICs
        self.encoder_jacob = catch_input(rom_dict, "encoder_jacob", False)
        self.encoder = None
        if self.encoder_jacob or (rom_domain.low_dim_init_files[model_idx] == ""):

            encoder_files = rom_dict["encoder_files"]
            assert len(encoder_files) == rom_domain.num_models, "Must provide encoder_files for each model"
            encoder_path = os.path.join(rom_domain.model_dir, encoder_files[model_idx])
            assert os.path.isfile(encoder_path), "Could not find encoder file at " + encoder_path
            self.encoder = self.load_model_obj(encoder_path)
            self.encoder_io_dtypes = self.check_model(decoder=False)

        # numerical Jacobian params
        self.numerical_jacob = catch_input(rom_dict, "numerical_jacob", False)
        self.fd_step = catch_input(rom_dict, "fd_step", FD_STEP_DEFAULT)

    def encode_sol(self, sol_in):
        """Compute full encoding of solution, including centering and normalization.

        Centers and normalizes full-dimensional sol_in, and then maps to low-dimensional code.
        Note that the apply_encoder is implemented within child classes, as these are specific to a specific library.

        Args:
            sol_in: Full-dimensional state to be encoded.

        Returns:
            Low-dimensional code NumPy array resulting from scaling and decoding.
        """

        sol = sol_in.copy()

        if self.target_cons:
            sol = self.scale_profile(
                sol,
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_cons,
                norm_sub_prof=self.norm_sub_prof_cons,
                center=True,
                cent_prof=self.cent_prof_cons,
                inverse=False,
            )

        else:
            sol = self.scale_profile(
                sol,
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_prim,
                norm_sub_prof=self.norm_sub_prof_prim,
                center=True,
                cent_prof=self.cent_prof_prim,
                inverse=False,
            )

        code = self.apply_encoder(sol)

        return code

    def init_from_sol(self, sol_domain):
        """Initialize full-order solution from encoding and decoding of loaded full-order initial conditions.

        Computes encoding and decoding of full-dimensional initial conditions and sets as
        new initial condition solution profile. This is automatically used if a low-dimensional state
        initial condition file is not provided.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
        """

        sol_int = sol_domain.sol_int

        if self.target_cons:
            self.code = self.encode_sol(sol_int.sol_cons[self.var_idxs, :])
            sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)
        else:
            self.code = self.encode_sol(sol_int.sol_prim[self.var_idxs, :])
            sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)

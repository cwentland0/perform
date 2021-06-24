import os

import numpy as np

from perform.constants import FD_STEP_DEFAULT
from perform.rom.projection_rom.projection_rom import ProjectionROM
from perform.input_funcs import catch_input


class AutoencoderProjROM(ProjectionROM):
    """Base class for all non-linear manifold projection-based ROMs using autoencoders.

    Inherits from ProjectionROM. Assumes that solution decoding is computed by

    sol = cent_prof + norm_sub_prof + norm_fac_prof * decoder(code)

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        mllib: MLLibrary instance associated with rom_domain.
        decoder:
            Decoder model object (e.g. tf.keras.Model) which maps from the low-dimensional state
            to the full-dimensional state.
        decoder_isconv: Boolean flag indicating whether decoder output layer is convolutional.
        decoder_io_format:
            Either "channels_first" (i.e. NCHW) or "channels_last" (i.e. NHWC) format of decoder output
            convolutional layer, if decoder_isconv = True. 
        decoder_io_dtypes: List containing the data type of the decoder input and output.
        decoder_io_shapes: tuple containing the tuple shapes of the decoder input and output.
        encoder_jacob:
            Boolean flag indicating whether the method should use an encoder Jacobian approximation, if applicable.
        encoder:
            Encoder model object (e.g. tf.keras.Model) which maps from the full-dimensional state
            to the low-dimensional state. Only required if initializing the simulation from a full-dimensional
            initial conditions or the ROM method requires an encoder.
        encoder_isconv: Boolean flag indicating whether encoder input layer is convolutional.
        encoder_io_format:
            Either "channels_first" (i.e. NCHW) or "channels_last" (i.e. NHWC) format of encoder input
            convolutional layer, if decoder_isconv = True. 
        encoder_io_dtypes: List containing the data type of the encoder input and output.
        encoder_io_shapes: tuple containing the tuple shapes of the encoder input and output.
        numerical_jacob:
            Boolean flag indicating whether the decoder/encoder Jacobian should be computed numerically.
            If False (default), computes the Jacobian analytically using automatic differentiation.
        fd_step: Finite-difference step size for computing the numerical decoder/encoder Jacobian, if requested.
        jacob_input: Persistent memory variable which will contain model inputs about which Jacobians are calculated.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)

        rom_dict = rom_domain.rom_dict
        self.mllib = rom_domain.mllib

        # Load decoder
        decoder_path = os.path.join(rom_domain.model_dir, rom_domain.model_files[model_idx])
        assert os.path.isfile(decoder_path), "Could not find decoder file at " + decoder_path
        self.decoder = self.mllib.load_model_obj(decoder_path)

        # Check I/O formatting
        self.decoder_isconv = catch_input(rom_dict, "model_isconv", False)
        self.decoder_io_format = catch_input(rom_dict, "model_io_format", None)
        decoder_input_shape = (self.latent_dim,)
        if self.decoder_isconv:
            if self.decoder_io_format == "channels_last":
                decoder_output_shape = (
                    self.num_cells,
                    self.num_vars,
                )
            elif self.decoder_io_format == "channels_first":
                decoder_output_shape = (
                    self.num_vars,
                    self.num_cells,
                )
            else:
                raise ValueError('Must specify model_io_format as "channels_first" or "channels_last" if model_isconv = True')
        else:
            decoder_output_shape = (self.num_cells * self.num_vars,)
        self.decoder_io_shapes, self.decoder_io_dtypes = self.mllib.check_model_io(
            self.decoder, decoder_input_shape, decoder_output_shape, self.decoder_isconv, self.decoder_io_format
        )

        # If required, load encoder
        # Encoder is required for encoder Jacobian or initializing from projection of full ICs
        self.encoder_jacob = catch_input(rom_dict, "encoder_jacob", False)
        self.encoder = None
        if self.encoder_jacob or (rom_domain.low_dim_init_files[model_idx] == ""):

            encoder_files = rom_dict["encoder_files"]
            assert len(encoder_files) == rom_domain.num_models, "Must provide encoder_files for each model"
            encoder_path = os.path.join(rom_domain.model_dir, encoder_files[model_idx])
            assert os.path.isfile(encoder_path), "Could not find encoder file at " + encoder_path
            self.encoder = self.mllib.load_model_obj(encoder_path)

            # Check I/O formatting
            self.encoder_isconv = catch_input(rom_dict, "encoder_isconv", False)
            self.encoder_io_format = catch_input(rom_dict, "encoder_io_format", None)
            encoder_output_shape = (self.latent_dim,)
            if self.encoder_isconv:
                if self.encoder_io_format == "channels_last":
                    encoder_input_shape = (
                        self.num_cells,
                        self.num_vars,
                    )
                elif self.encoder_io_format == "channels_first":
                    encoder_input_shape = (
                        self.num_vars,
                        self.num_cells,
                    )
                else:
                    raise ValueError('Must specify encoder_io_format as "channels_first" or "channels_last" if encoder_isconv = True')
            else:
                encoder_input_shape = (self.num_cells * self.num_vars,)
            self.encoder_io_shapes, self.encoder_io_dtypes = self.mllib.check_model_io(
                self.encoder, encoder_input_shape, encoder_output_shape, self.encoder_isconv, self.encoder_io_format
            )

        # numerical Jacobian params
        self.numerical_jacob = catch_input(rom_dict, "numerical_jacob", False)
        self.fd_step = catch_input(rom_dict, "fd_step", FD_STEP_DEFAULT)

        # initialize persistent memory for ML model calculations
        self.jacob_input = None
        if not self.numerical_jacob:
            if self.encoder_jacob:
                self.jacob_input = self.mllib.init_persistent_mem(
                    self.encoder_io_shapes[0], dtype=self.encoder_io_dtypes[0], prepend_batch=True
                )
            else:
                self.jacob_input = self.mllib.init_persistent_mem(
                    self.decoder_io_shapes[0], dtype=self.decoder_io_dtypes[0], prepend_batch=True
                )

    def apply_decoder(self, code):
        """Compute raw decoding of code.

        Only computes decoder(code), does not compute any denormalization or decentering.

        Args:
            code: NumPy array of low-dimensional state, of dimension latent_dim.

        Returns:
            NumPy array of decoded full-dimensional state.
        """

        sol = self.mllib.infer_model(self.decoder, code)
        if self.decoder_io_format == "channels_last":
            sol = sol.T

        return sol

    def apply_encoder(self, sol):
        """Compute raw encoding of full-dimensional state.

        Only computes encoder(sol), does not compute any centering or normalization.

        Args:
            sol: NumPy array of full-dimensional state.

        Returns:
            NumPy array of encoded latent variables.
        """

        if self.encoder_io_format == "channels_last":
            sol_in = (sol.copy()).T
        else:
            sol_in = sol.copy()
        code = self.mllib.infer_model(self.encoder, sol_in)

        return code

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
        elif self.target_prim:
            sol = self.scale_profile(
                sol,
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_prim,
                norm_sub_prof=self.norm_sub_prof_prim,
                center=True,
                cent_prof=self.cent_prof_prim,
                inverse=False,
            )
        else:
            raise ValueError("Invalid variable mapping.")

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
        elif self.target_prim:
            self.code = self.encode_sol(sol_int.sol_prim[self.var_idxs, :])
            sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)
        else:
            raise ValueError("Invalid variable mapping.")

    def calc_jacobian(self, sol_domain, encoder_jacob=False):
        """Calculate decoder or encoder Jacobian.

        Computes numerical or analytical Jacobian of decoder with respect to current low-dimensional state
        or of encoder with respect to current full-dimensional conservative state.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
            encoder_jacob: Boolean flag indicating whether the encoder Jacobian should be computed
        
        Returns:
            NumPy array of encoder or decoder Jacobian, reshaped for use with implicit time-integration.
        """

        if encoder_jacob:
            # TODO: only calculate standardized solution once, hang onto it
            # Don't have to pass sol_domain, too

            # get input
            if self.target_cons:
                model_input = self.scale_profile(
                    sol_domain.sol_int.sol_cons[self.var_idxs, :],
                    normalize=True,
                    norm_fac_prof=self.norm_fac_prof_cons,
                    norm_sub_prof=self.norm_sub_prof_cons,
                    center=True,
                    centProf=self.cent_prof_cons,
                    inverse=False,
                )
            elif self.target_prim:
                model_input = self.scale_profile(
                    sol_domain.sol_int.sol_prim[self.var_idxs, :],
                    normalize=True,
                    norm_fac_prof=self.norm_fac_prof_prim,
                    norm_sub_prof=self.norm_sub_prof_prim,
                    center=True,
                    centProf=self.cent_prof_prim,
                    inverse=False,
                )
            else:
                raise ValueError("Invalid variable mapping.")

            if self.encoder_io_format == "channels_last":
                model_input = np.transpose(model_input, axes=(1, 0))

            # calculate Jacobian and reshape
            jacob = self.mllib.calc_model_jacobian(
                self.encoder,
                model_input,
                self.encoder_io_shapes[1],
                numerical=self.numerical_jacob,
                fd_step=self.fd_step,
                persistent_input=self.jacob_input,
            )
            if self.encoder_isconv:
                if self.encoder_io_format == "channels_last":
                    jacob = np.transpose(jacob, axes=(0, 2, 1))
            jacob = np.reshape(jacob, (self.latent_dim, -1), order="C")

        else:

            # calculate Jacobian and reshape
            jacob = self.mllib.calc_model_jacobian(
                self.decoder,
                self.code,
                self.decoder_io_shapes[1],
                numerical=self.numerical_jacob,
                fd_step=self.fd_step,
                persistent_input=self.jacob_input,
            )
            if self.decoder_isconv:
                if self.decoder_io_format == "channels_last":
                    jacob = np.transpose(jacob, axes=(1, 0, 2))
            jacob = np.reshape(jacob, (-1, self.latent_dim), order="C")

        return jacob

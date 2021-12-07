import os

import numpy as np
from scipy.linalg import pinv

from perform.input_funcs import catch_input
from perform.rom.rom_space_mapping.rom_space_mapping import RomSpaceMapping


class AutoencoderSpaceMapping(RomSpaceMapping):
    """Autoencoder mapping to/from the state/latent spaces"""

    def __init__(self, sol_domain, rom_domain, rom_model):

        rom_dict = rom_domain.rom_dict
        assert rom_domain.ml_library != "none"
        self.mllib = rom_domain.mllib

        # Decoder input checking
        decoder_files = rom_dict["decoder_files"]
        assert len(decoder_files) == rom_domain.num_models, "Must provide decoder_files for each model"  # redundant
        in_file = os.path.join(rom_domain.model_dir, decoder_files[rom_model.model_idx])
        assert os.path.isfile(in_file), "Could not find decoder file at " + in_file
        self.decoder_file = in_file
        self.decoder_isconv = catch_input(rom_dict, "decoder_isconv", False)
        self.decoder_io_format = catch_input(rom_dict, "model_io_format", None)

        # If required, encoder input checking
        self.encoder_file = None
        if rom_domain.code_init_files[rom_model.model_idx] == "":
            encoder_files = rom_dict["encoder_files"]
            assert len(encoder_files) == rom_domain.num_models, "Must provide encoder_files for each model"
            in_file = os.path.join(rom_domain.model_dir, encoder_files[rom_model.model_idx])
            assert os.path.isfile(in_file), "Could not find encoder file at " + in_file
            self.encoder_file = in_file
            self.encoder_isconv = catch_input(rom_dict, "encoder_isconv", False)
            self.encoder_io_format = catch_input(rom_dict, "encoder_io_format", None)

        super().__init__(sol_domain, rom_domain, rom_model)

    def load_mapping(self):

        # Load decoder and check I/O formatting
        self.decoder = self.mllib.load_model_obj(self.decoder_file)
        decoder_input_shape = (self.latent_dim,)
        if self.decoder_isconv:
            if self.decoder_io_format == "channels_last":
                decoder_output_shape = (
                    self.sol_shape[1],
                    self.sol_shape[0],
                )
            elif self.decoder_io_format == "channels_first":
                decoder_output_shape = (
                    self.sol_shape[0],
                    self.sol_shape[1],
                )
            else:
                raise ValueError(
                    'Must specify model_io_format as "channels_first" or "channels_last" if model_isconv = True'
                )
        else:
            decoder_output_shape = (self.sol_shape[0] * self.sol_shape[1],)

        self.decoder_io_shapes, self.decoder_io_dtypes = self.mllib.check_model_io(
            self.decoder, decoder_input_shape, decoder_output_shape, self.decoder_isconv, self.decoder_io_format
        )

        # presistent memory for ML model calculations
        self.jacob_input = self.mllib.init_persistent_mem(
            self.decoder_io_shapes[0], dtype=self.decoder_io_dtypes[0], prepend_batch=True
        )

        # If required, load encoder and check I/O formatting
        if self.encoder_file is not None:
            self.encoder = self.mllib.load_model_obj(self.encoder_file)
            encoder_output_shape = (self.latent_dim,)
            if self.encoder_isconv:
                if self.encoder_io_format == "channels_last":
                    encoder_input_shape = (
                        self.sol_shape[1],
                        self.sol_shape[0],
                    )
                elif self.encoder_io_format == "channels_first":
                    encoder_input_shape = (
                        self.sol_shape[0],
                        self.sol_shape[1],
                    )
                else:
                    raise ValueError(
                        'Must specify encoder_io_format as "channels_first" or "channels_last" if encoder_isconv = True'
                    )
            else:
                encoder_input_shape = (self.sol_shape[0] * self.sol_shape[1],)
            self.encoder_io_shapes, self.encoder_io_dtypes = self.mllib.check_model_io(
                self.encoder, encoder_input_shape, encoder_output_shape, self.encoder_isconv, self.encoder_io_format
            )

    def apply_encoder(self, sol):

        if self.encoder_isconv:
            if self.encoder_io_format == "channels_last":
                sol_in = (sol.copy()).T
            else:
                sol_in = sol.copy()
        else:
            sol_in = sol.ravel(order="C")

        code = self.mllib.infer_model(self.encoder, sol_in)

        return code

    def apply_decoder(self, code):

        sol = self.mllib.infer_model(self.decoder, code)

        if self.decoder_isconv:
            if self.decoder_io_format == "channels_last":
                sol = sol.T
        else:
            sol = np.reshape(sol, self.sol_shape)

        return sol

    def calc_decoder_jacob_pinv(self, code, jacob=None):

        if jacob is None:
            jacob = self.calc_decoder_jacob(code)
        projector = pinv(jacob)

        return projector

    def calc_decoder_jacob(self, code):
        """Calculate decoder Jacobian.

        Computes numerical or analytical Jacobian of decoder with respect to current low-dimensional state
        or of encoder with respect to current full-dimensional conservative state.

        Args:
            encoder_jacob: Boolean flag indicating whether the encoder Jacobian should be computed

        Returns:
            NumPy array of encoder or decoder Jacobian, reshaped for use with implicit time-integration.
        """

        # calculate Jacobian and reshape
        jacob = self.mllib.calc_model_jacobian(
            self.decoder,
            code,
            self.decoder_io_shapes[1],
            persistent_input=self.jacob_input,
        )
        if self.decoder_isconv:
            if self.decoder_io_format == "channels_last":
                jacob = np.transpose(jacob, axes=(1, 0, 2))
        jacob = np.reshape(jacob, (-1, self.latent_dim), order="C")

        return jacob

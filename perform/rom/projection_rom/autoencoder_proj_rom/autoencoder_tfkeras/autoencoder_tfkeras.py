import numpy as np
import tensorflow as tf

from perform.constants import REAL_TYPE
from perform.input_funcs import catch_input
from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_proj_rom import AutoencoderProjROM
from perform.rom.tf_keras_funcs import (
    init_device, load_model_obj, get_io_shape)


class AutoencoderTFKeras(AutoencoderProjROM):
    """
    Base class for autoencoder projection-based ROMs using TensorFlow-Keras

    See user guide for notes on expected input format
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        self.run_gpu = catch_input(rom_domain.rom_dict, "run_gpu", False)
        init_device(self.run_gpu)

        # Store function object for use in parent routines
        self.load_model_obj = load_model_obj

        # "nchw" (channels first) or "nhwc" (channels last)
        self.io_format = rom_domain.rom_dict["io_format"]
        if self.io_format == "nchw":
            assert (self.run_gpu), "Tensorflow cannot handle NCHW on CPUs"
        elif self.io_format == "nhwc":
            pass  # works on GPU or CPU
        else:
            raise ValueError("io_format must be either \"nchw\" or \"nhwc\";"
                             + "you entered " + str(self.io_format))

        super().__init__(model_idx, rom_domain, sol_domain)

        # Initialize tf.Variable for Jacobian calculations
        # Otherwise, recreating this will cause retracing
        # of the computational graph
        sol_int = sol_domain.sol_int
        if not self.numerical_jacob:
            if self.encoder_jacob:
                if self.io_format == "nhwc":
                    if self.target_cons:
                        sol_init = (
                            (sol_int.sol_cons.T)[None, :, self.var_idxs])
                    else:
                        sol_init = (
                            (sol_int.sol_prim.T)[None, :, self.var_idxs])
                else:
                    if self.target_cons:
                        sol_init = (
                            (sol_int.sol_cons)[None, :, self.var_idxs])
                    else:
                        sol_init = (
                            (sol_int.sol_prim)[None, :, self.var_idxs])
                self.jacob_input = tf.Variable(
                    sol_init, dtype=self.encoder_io_dtypes[0])

            else:
                self.jacob_input = tf.Variable(
                    self.code[None, :], dtype=self.decoder_io_dtypes[0])

    def check_model(self, decoder=True):
        """
        Check decoder/encoder input/output dimensions and returns I/O dtypes
        """

        if decoder:
            input_shape = get_io_shape(self.decoder.layers[0].input_shape)
            output_shape = get_io_shape(self.decoder.layers[-1].output_shape)

            assert(input_shape[-1] == self.latent_dim), (
                "Mismatched decoder input shape: "
                + str(input_shape[-1]) + ", " + str(self.latent_dim))

            if self.io_format == "nchw":
                assert(output_shape[-2:] == self.sol_shape), (
                    "Mismatched decoder output shape: "
                    + str(output_shape[-2:]) + ", " + str(self.sol_shape))

            else:
                assert(output_shape[-2:] == self.sol_shape[::-1]), (
                    "Mismatched decoder output shape: "
                    + str(output_shape[-2:]) + ", "
                    + str(self.sol_shape[::-1]))

            input_dtype = self.decoder.layers[0].dtype
            output_dtype = self.decoder.layers[-1].dtype

        else:
            input_shape = get_io_shape(self.encoder.layers[0].input_shape)
            output_shape = get_io_shape(self.encoder.layers[-1].output_shape)

            assert(output_shape[-1] == self.latent_dim), (
                "Mismatched encoder output shape: "
                + str(output_shape[-1]) + ", " + str(self.latent_dim))

            if self.io_format == "nchw":
                assert(input_shape[-2:] == self.sol_shape), (
                    "Mismatched encoder output shape: "
                    + str(input_shape[-2:]) + ", " + str(self.sol_shape))
            else:
                assert(input_shape[-2:] == self.sol_shape[::-1]), (
                    "Mismatched encoder output shape: "
                    + str(input_shape[-2:]) + ", "
                    + str(self.sol_shape[::-1]))

            input_dtype = self.encoder.layers[0].dtype
            output_dtype = self.encoder.layers[-1].dtype

        return [input_dtype, output_dtype]

    def apply_decoder(self, code):
        """
        Compute raw decoding of code, without de-normalizing or de-centering
        """

        sol = np.squeeze(self.decoder(code[None, :]).numpy(), axis=0)
        if self.io_format == "nhwc":
            sol = sol.T

        return sol

    def apply_encoder(self, sol):
        """
        Compute raw encoding of solution,
        assuming it has been centered and normalized
        """

        if self.io_format == "nhwc":
            sol_in = (sol.copy()).T
        else:
            sol_in = sol.copy()
        code = np.squeeze(self.encoder(sol_in[None, :, :]).numpy(), axis=0)

        return code

    @tf.function
    def calc_analytical_model_jacobian(self, model, inputs):
        """
        Compute analytical Jacobian of TensorFlow-Keras model
        using GradientTape

        NOTE: inputs is a tf.Variable
        """

        with tf.GradientTape() as g:
            outputs = model(inputs)
        jacob = g.jacobian(outputs, inputs)

        return jacob

    def calc_numerical_model_jacobian(self, model, inputs):
        """
        Compute numerical Jacobian of TensorFlow-Keras models
        by finite difference approximation

        NOTE: inputs is a np.ndarray
        """

        # TODO: implement encoder Jacobian

        if self.encoder_jacob:
            raise ValueError("Numerical encoder Jacobian not implemented yet")
            if self.io_format == "nhwc":
                jacob = np.zeros(
                    (inputs.shape[0], self.num_cells, self.numVars),
                    dtype=REAL_TYPE)
            else:
                jacob = np.zeros(
                    (inputs.shape[0], self.numVars, self.num_cells),
                    dtype=REAL_TYPE)

        else:
            if self.io_format == "nhwc":
                jacob = np.zeros(
                    (self.num_cells, self.numVars, inputs.shape[0]),
                    dtype=REAL_TYPE)
            else:
                jacob = np.zeros(
                    (self.numVars, self.num_cells, inputs.shape[0]),
                    dtype=REAL_TYPE)

        # get initial prediction
        outputsBase = np.squeeze(model(inputs[None, :]).numpy(), axis=0)

        # TODO: this does not generalize to encoder Jacobian
        for elemIdx in range(0, inputs.shape[0]):

            # perturb
            inputsPert = inputs.copy()
            inputsPert[elemIdx] = inputsPert[elemIdx] + self.fd_step

            # make prediction at perturbed state
            outputs = np.squeeze(model(inputsPert[None, :]).numpy(), axis=0)

            # compute finite difference approximation
            jacob[:, :, elemIdx] = (outputs - outputsBase) / self.fd_step

        return jacob

    def calc_model_jacobian(self, sol_domain):
        """
        Helper function for calculating TensorFlow-Keras model Jacobian
        """

        # TODO: generalize this for generic model, input

        if self.encoder_jacob:
            # TODO: only calculate standardized solution once, hang onto it
            # Don't have to pass sol_domain, too
            sol = self.standardize_data(
                sol_domain.sol_int.sol_cons[self.var_idxs, :],
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_cons,
                norm_sub_prof=self.normSubProfCons,
                center=True, centProf=self.centProfCons,
                inverse=False)

            if self.io_format == "nhwc":
                sol = np.transpose(sol, axes=(1, 0))

            if self.numerical_jacob:
                jacob = self.calc_numerical_model_jacobian(
                    self.encoder, sol)

            else:
                self.jacob_input.assign(sol[None, :, :])
                jacob_tf = self.calc_analytical_model_jacobian(
                    self.encoder, self.jacob_input)
                jacob = tf.squeeze(jacob_tf, axis=[0, 2]).numpy()

            if self.io_format == "nhwc":
                jacob = np.transpose(jacob, axes=(0, 2, 1))

            jacob = np.reshape(jacob, (self.latent_dim, -1), order='C')

        else:

            if self.numerical_jacob:
                jacob = self.calc_numerical_model_jacobian(
                    self.decoder, self.code)
            else:
                self.jacob_input.assign(self.code[None, :])
                jacob_tf = self.calc_analytical_model_jacobian(
                    self.decoder, self.jacob_input)
                jacob = tf.squeeze(jacob_tf, axis=[0, 3]).numpy()

            if self.io_format == "nhwc":
                jacob = np.transpose(jacob, axes=(1, 0, 2))

            jacob = np.reshape(jacob, (-1, self.latent_dim), order='C')

        return jacob

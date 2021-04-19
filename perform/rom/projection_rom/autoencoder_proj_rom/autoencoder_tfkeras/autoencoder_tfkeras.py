from packaging import version
from time import sleep

import numpy as np
import tensorflow as tf

from perform.constants import REAL_TYPE
from perform.input_funcs import catch_input
from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_proj_rom import AutoencoderProjROM
from perform.rom.tf_keras_funcs import init_device, load_model_obj, get_io_shape


class AutoencoderTFKeras(AutoencoderProjROM):
    """Base class for autoencoder projection-based ROMs using TensorFlow-Keras.

    Inherits from AutoencoderProjROM. Supplies library-specific functions noted in AutoencoderProjROM.
    
    Child classes must implement a calc_projector() member function if it permits explicit time integration,
    and/or a calc_d_code() member function if it permits implicit time integration.

    It is expected that models supplied are saved in the older Keras H5 format.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        load_model_obj:
            Function supplied to parent AutoencoderProjROM to load TF-Keras model files,
            found in tf_keras_funcs.py.
        io_format:
            Either "nchw" (i.e. "channels-first" in Keras) or "nhwc" (i.e. "channels-last in Keras),
            format of model input and output. Note that TF-Keras cannot compute convolutional layers on the CPU
            (run_gpu=False) if io_format="nchw"; an error is thrown if this is the case.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if version.parse(tf.__version__) < version.parse("2.4.1"):
            print("WARNING: You are using TensorFlow version < 2.4.1, proper ROM behavior not guaranteed")
            sleep(1.0)

        run_gpu = catch_input(rom_domain.rom_dict, "run_gpu", False)
        init_device(run_gpu)

        # Store function object for use in parent routines
        self.load_model_obj = load_model_obj

        # "nchw" (channels first) or "nhwc" (channels last)
        self.io_format = rom_domain.rom_dict["io_format"]
        if self.io_format == "nchw":
            assert run_gpu, "Tensorflow cannot handle NCHW on CPUs"
        elif self.io_format == "nhwc":
            pass  # works on GPU or CPU
        else:
            raise ValueError('io_format must be either "nchw" or "nhwc"; you entered ' + str(self.io_format))

        super().__init__(model_idx, rom_domain, sol_domain)

        # Initialize tf.Variable for Jacobian calculations
        # Otherwise, recreating this will cause retracing of the computational graph
        sol_int = sol_domain.sol_int
        if not self.numerical_jacob:
            if self.encoder_jacob:
                if self.io_format == "nhwc":
                    if self.target_cons:
                        sol_init = (sol_int.sol_cons.T)[None, :, self.var_idxs]
                    else:
                        sol_init = (sol_int.sol_prim.T)[None, :, self.var_idxs]
                else:
                    if self.target_cons:
                        sol_init = (sol_int.sol_cons)[None, :, self.var_idxs]
                    else:
                        sol_init = (sol_int.sol_prim)[None, :, self.var_idxs]
                self.jacob_input = tf.Variable(sol_init, dtype=self.encoder_io_dtypes[0])

            else:
                self.jacob_input = tf.Variable(self.code[None, :], dtype=self.decoder_io_dtypes[0])

    def check_model(self, decoder=True):
        """Check decoder/encoder input/output dimensions and returns I/O dtypes

        Extracts shapes of this model's decoder or encoder and checks whether they match the expected shapes,
        as determined by num_vars, num_cells, and latent_dim.

        Args:
            decoder: 
                Boolean indicating whether to return the shape of this RomModel's decoder (decoder=True)
                or encoder (decoder=False).

        Returns:
            A list containing two entries: the tuple shape of the model input, and the tuple shape of the model output.
        """

        if decoder:
            input_shape = get_io_shape(self.decoder.layers[0].input_shape)
            output_shape = get_io_shape(self.decoder.layers[-1].output_shape)

            assert input_shape[-1] == self.latent_dim, (
                "Mismatched decoder input shape: " + str(input_shape[-1]) + ", " + str(self.latent_dim)
            )

            if self.io_format == "nchw":
                assert output_shape[-2:] == self.sol_shape, (
                    "Mismatched decoder output shape: " + str(output_shape[-2:]) + ", " + str(self.sol_shape)
                )

            else:
                assert output_shape[-2:] == self.sol_shape[::-1], (
                    "Mismatched decoder output shape: " + str(output_shape[-2:]) + ", " + str(self.sol_shape[::-1])
                )

            input_dtype = self.decoder.layers[0].dtype
            output_dtype = self.decoder.layers[-1].dtype

        else:
            input_shape = get_io_shape(self.encoder.layers[0].input_shape)
            output_shape = get_io_shape(self.encoder.layers[-1].output_shape)

            assert output_shape[-1] == self.latent_dim, (
                "Mismatched encoder output shape: " + str(output_shape[-1]) + ", " + str(self.latent_dim)
            )

            if self.io_format == "nchw":
                assert input_shape[-2:] == self.sol_shape, (
                    "Mismatched encoder output shape: " + str(input_shape[-2:]) + ", " + str(self.sol_shape)
                )
            else:
                assert input_shape[-2:] == self.sol_shape[::-1], (
                    "Mismatched encoder output shape: " + str(input_shape[-2:]) + ", " + str(self.sol_shape[::-1])
                )

            input_dtype = self.encoder.layers[0].dtype
            output_dtype = self.encoder.layers[-1].dtype

        return [input_dtype, output_dtype]

    def apply_decoder(self, code):
        """Compute raw decoding of code.
        
        Only computes decoder(code), does not compute any denormalization or decentering.

        Args:
            code: NumPy array of low-dimensional state, of dimension latent_dim.
        """

        sol = np.squeeze(self.decoder(code[None, :]).numpy(), axis=0)
        if self.io_format == "nhwc":
            sol = sol.T

        return sol

    def apply_encoder(self, sol):
        """Compute raw encoding of full-dimensional state.
        
        Only computes encoder(sol), does not compute any centering or normalization.

        Args:
            sol: NumPy array of full-dimensional state.
        """

        if self.io_format == "nhwc":
            sol_in = (sol.copy()).T
        else:
            sol_in = sol.copy()
        code = np.squeeze(self.encoder(sol_in[None, :, :]).numpy(), axis=0)

        return code

    @tf.function
    def calc_analytical_model_jacobian(self, model, inputs):
        """Compute analytical Jacobian of TensorFlow-Keras model using GradientTape.

        Calculates the analytical Jacobian of an encoder or decoder with respect to the given inputs.
        The GradientTape method computes this using automatic differentiation.

        Args:
            model: tf.keras.Model for which the analytical Jacobian should be computed.
            inputs: tf.Variable containing inputs to decoder about which the model Jacobian should be computed.

        Returns:
            tf.Variable containing the analytical Jacobian, without squeezing any singleton dimensions.
        """

        with tf.GradientTape() as g:
            outputs = model(inputs)
        jacob = g.jacobian(outputs, inputs)

        return jacob

    def calc_numerical_model_jacobian(self, model, inputs):
        """Compute numerical Jacobian of TensorFlow-Keras model using finite-difference.

        Calculates the numerical Jacobian of an encoder or decoder with respect to the given inputs.
        A finite-difference approximation of the gradient with respect to each element of inputs is calculated.
        The fd_step attribute determines the finite difference step size.

        Args:
            model: tf.keras.Model for which the numerical Jacobian should be computed.
            inputs: NumPy array containing inputs to decoder about which the model Jacobian should be computed.

        Returns:
            NumPy array containing the numerical Jacobian.
        """

        # TODO: implement encoder Jacobian

        if self.encoder_jacob:
            raise ValueError("Numerical encoder Jacobian not implemented yet")
            if self.io_format == "nhwc":
                jacob = np.zeros((inputs.shape[0], self.num_cells, self.numVars), dtype=REAL_TYPE)
            else:
                jacob = np.zeros((inputs.shape[0], self.numVars, self.num_cells), dtype=REAL_TYPE)

        else:
            if self.io_format == "nhwc":
                jacob = np.zeros((self.num_cells, self.numVars, inputs.shape[0]), dtype=REAL_TYPE)
            else:
                jacob = np.zeros((self.numVars, self.num_cells, inputs.shape[0]), dtype=REAL_TYPE)

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
        """Helper function for calculating TensorFlow-Keras model Jacobian

        Computes analytical or numerical Jacobian of a decoder or encoder, depending on the requested
        ROM solution method. Handles the various data formats and array shapes produced by each option,
        and returns the properly-formatted model Jacobian to the child classes calling this function.

        Args:
            sol_domain: olutionDomain with which this RomModel's RomDomain is associated.

        Returns:
            NumPy array of model Jacobian, formatted appropriately for time integration.
        """

        # TODO: generalize this for generic model, input

        if self.encoder_jacob:
            # TODO: only calculate standardized solution once, hang onto it
            # Don't have to pass sol_domain, too
            sol = self.scale_profile(
                sol_domain.sol_int.sol_cons[self.var_idxs, :],
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_cons,
                norm_sub_prof=self.normSubProfCons,
                center=True,
                centProf=self.centProfCons,
                inverse=False,
            )

            if self.io_format == "nhwc":
                sol = np.transpose(sol, axes=(1, 0))

            if self.numerical_jacob:
                jacob = self.calc_numerical_model_jacobian(self.encoder, sol)

            else:
                self.jacob_input.assign(sol[None, :, :])
                jacob_tf = self.calc_analytical_model_jacobian(self.encoder, self.jacob_input)
                jacob = tf.squeeze(jacob_tf, axis=[0, 2]).numpy()

            if self.io_format == "nhwc":
                jacob = np.transpose(jacob, axes=(0, 2, 1))

            jacob = np.reshape(jacob, (self.latent_dim, -1), order="C")

        else:

            if self.numerical_jacob:
                jacob = self.calc_numerical_model_jacobian(self.decoder, self.code)
            else:
                self.jacob_input.assign(self.code[None, :])
                jacob_tf = self.calc_analytical_model_jacobian(self.decoder, self.jacob_input)
                jacob = tf.squeeze(jacob_tf, axis=[0, 3]).numpy()

            if self.io_format == "nhwc":
                jacob = np.transpose(jacob, axes=(1, 0, 2))

            jacob = np.reshape(jacob, (-1, self.latent_dim), order="C")

        return jacob

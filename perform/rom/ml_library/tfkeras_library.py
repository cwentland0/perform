import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from perform.rom.ml_library.ml_library import MLLibrary
from perform.constants import FD_STEP_DEFAULT


class TFKerasLibrary(MLLibrary):
    """Class for accessing Tensorflow-Keras functionalities."""

    def __init__(self, rom_domain):

        super().__init__(rom_domain)

    def init_device(self, run_gpu):
        """Initializes GPU execution, if requested

        TensorFlow 2+ can execute from CPU or GPU, this function does some prep work.

        Passing run_gpu=True will limit GPU memory growth, as unlimited TensorFlow memory allocation can be
        quite aggressive on first call.

        Passing run_gpu=False will guarantee that TensorFlow runs on the CPU. Even if GPUs are available,
        this will hide those devices from the TensorFlow runtime.

        Args:
            run_gpu: Boolean flag indicating whether to execute TensorFlow functions on an available GPU.
        """

        if run_gpu:
            # make sure TF doesn't gobble up device memory
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            # If GPU is available, TF will automatically run there
            # 	This forces to run on CPU even if GPU is available
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def load_model_obj(self, model_path, custom_objects=None):
        """Load model object from file.

        This function loads a trained Tensorflow-Keras model.
        Either the newer SavedModel

        Args:
            model_path: string path to the model file to be loaded.

        Returns:
            Uncompiled Keras model object. As no training is being done, there is no need to compile.
        """

        model_obj = load_model(model_path, compile=False, custom_objects=custom_objects)

        return model_obj

    def init_persistent_mem(self, input_shape, input_dtype="float64", prepend_batch=False):
        """Initialize persistent memory for graph computations.

        In order to use tf.function for efficient graph computations, some persistent memory must be allocated.

        Args:
            model: child class of RomModel which provides necessary Boolean flags.
            model_type: string denoting the appropriate types of memory to allocate for a given model.
            sol_domain: SolutionDomain containing physical solutions.
            code: low-dimensional latent variables.

        Returns:
            Tuple of presistent memory variables, specific to the requested model_type.
        """

        mem_input = np.zeros(input_shape, dtype=input_dtype)
        if prepend_batch:
            mem_input = np.expand_dims(mem_input, axis=0)
        mem_obj = tf.Variable(mem_input, dtype=input_dtype)

        return mem_obj

    def check_conv_io_format(self, io_format):
        """Checks for convolutional model I/O compatability."""

        # "nchw" (channels first) or "nhwc" (channels last)
        if io_format == "nchw":
            assert self.run_gpu, "Tensorflow cannot handle NCHW on CPUs"
        elif io_format == "nhwc":
            pass  # works on GPU or CPU
        else:
            raise ValueError(
                'io_format for a convolutional model must be either "nchw" or "nhwc"; you entered ' + str(io_format)
            )

    def get_io_shape(self, model):
        """Gets model I/O shape, excluding batch dimension.

        

        Args:
            shape: output of tf.keras.Model.Layer.input_shape or tf.keras.Model.Layer.output_shape
        """

        input_shape = self.get_shape_tuple(model.layers[0].input_shape)[1:]
        output_shape = self.get_shape_tuple(model.layers[-1].output_shape)[1:]

        return (input_shape, output_shape)

    def get_io_dtype(self, model):

        return [model.layers[0].dtype, model.layers[-1].dtype]

    def infer_model(self, model, inputs):
        """Compute inference of model.

        Args:
            model: Keras model object.
            inputs: NumPy array of inputs.

        Returns:
            NumPy array of inferred output, with batch dimension squeezed.
        """

        inference = np.squeeze(model(np.expand_dims(inputs, axis=0)).numpy(), axis=0)

        return inference

    @tf.function
    def calc_analytical_model_jacobian(self, model, inputs):
        """Compute analytical Jacobian of model.

        Calculates the analytical Jacobian of a model with respect to the given inputs.
        The GradientTape method computes this using automatic differentiation.

        Args:
            model: tf.keras.Model for which the analytical Jacobian should be computed.
            inputs: tf.Variable containing inputs to model about which the model Jacobian should be computed.

        Returns:
            tf.Variable containing the analytical Jacobian, without squeezing any singleton dimensions.
        """

        with tf.GradientTape() as g:
            outputs = model(inputs)
        jacob = g.jacobian(outputs, inputs)

        return jacob

    def calc_model_jacobian(
        self, model, input, output_shape, numerical=False, fd_step=FD_STEP_DEFAULT, persistent_input=None
    ):
        """Helper function for calculating TensorFlow-Keras model Jacobian

        Computes analytical or numerical Jacobian of a decoder or encoder, depending on the requested
        ROM solution method. Handles the various data formats and array shapes produced by each option,
        and returns the properly-formatted model Jacobian to the child classes calling this function.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

        Returns:
            NumPy array of model Jacobian, formatted appropriately for time integration.
        """

        if numerical:
            jacob = self.calc_numerical_model_jacobian(model, input, output_shape, fd_step)
        else:
            persistent_input.assign(np.expand_dims(input, axis=0))
            jacob_tf = self.calc_analytical_model_jacobian(model, persistent_input)
            jacob = tf.squeeze(jacob_tf, axis=[0, len(output_shape) + 1]).numpy()

        return jacob

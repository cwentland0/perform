import os
from packaging import version

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from perform.rom.ml_library.ml_library import MLLibrary


class TFKerasLibrary(MLLibrary):
    """Class for implementing Tensorflow-Keras functionalities.

    This class assumes that Tensorflow >=2.0 is installed.

    Args:
        rom_domain: RomDomain which contains relevant ROM input dictionary.
    """

    def __init__(self, rom_domain):

        super().__init__(rom_domain)
        if version.parse(tf.__version__) < version.parse("2.0"):
            raise ValueError("You must use TensorFlow >=2.0, please upgrade your installation of Tensorflow.")

    def init_device(self, run_gpu):
        """Initializes GPU execution, if requested.

        TensorFlow >=2.0 can execute from CPU or GPU, this function does some prep work.

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
        Either the newer SavedModel format or old Keras HDF5 format are supported.

        Args:
            model_path: string path to the model file to be loaded.
            custom_objects: dictionary of custom classes or functions for use in model.

        Returns:
            Uncompiled tf.keras.Model object. As no training is being done, there is no need to compile.
        """

        model_obj = load_model(model_path, compile=False, custom_objects=custom_objects)

        return model_obj

    def init_persistent_mem(self, data_shape, dtype="float64", prepend_batch=False):
        """Initialize persistent memory for graph computations.

        In order to use tf.function for efficient graph computations, some persistent memory must be allocated.

        Args:
            data_shape: tuple shape of memory object to be created.
            dtype: data type of memory object to be created.
            prepend_batch: Boolean flag indicating whether to prepend a singleton batch dimension to memory object.

        Returns:
            tf.Variable object to be used for persistent memory.
        """

        mem_np = np.zeros(data_shape, dtype=dtype)
        if prepend_batch:
            mem_np = np.expand_dims(mem_np, axis=0)
        mem_obj = tf.Variable(mem_np, dtype=dtype)

        return mem_obj

    def check_conv_io_format(self, io_format):
        """Checks for convolutional model I/O compatability.

        Tensorflow >=2.0 cannot handle NCHW convolutional layers when running on a CPU.
        The user input is really only for handing input/output convolutional layers, if inner layers
        in a deep network use incompatible convolutional layers then execution will still fail on CPUs.

        Args:
            io_format: either "channels_first" or "channels_last", indicating convolutional layer format.
        """

        # "channels_first" or "channels_last"
        if io_format == "channels_first":
            assert self.run_gpu, "Tensorflow cannot handle channels_first on CPUs"
        elif io_format == "channels_last":
            pass  # works on GPU or CPU
        else:
            raise ValueError(
                'io_format for a convolutional model must be either "channels_first" or "channels_last"; you entered '
                + str(io_format)
            )

    def get_io_shape(self, model):
        """Gets model I/O shape, excluding batch dimension.

        Args:
            model: tf.keras.Model instance.

        Returns:
            Tuple shapes of model input and model output.
        """

        input_shape = self.get_shape_tuple(model.layers[0].input_shape)[1:]
        output_shape = self.get_shape_tuple(model.layers[-1].output_shape)[1:]

        return (input_shape, output_shape)

    def get_io_dtype(self, model):
        """Gets model I/O data types.

        Args:
            model: tf.keras.Model instance.

        Returns:
            List of model input and output data types.
        """

        return [model.layers[0].dtype, model.layers[-1].dtype]

    def infer_model(self, model, inputs):
        """Compute inference of model.

        Args:
            model: tf.keras.Model instance.
            inputs: NumPy array of inputs to model.

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
            model: tf.keras.Model instance.
            inputs: tf.Variable containing inputs to model about which the model Jacobian should be computed.

        Returns:
            tf.Variable containing the analytical Jacobian, without squeezing any singleton dimensions.
        """

        with tf.GradientTape() as g:
            outputs = model(inputs)
        jacob = g.jacobian(outputs, inputs)

        return jacob

    def calc_model_jacobian(self, model, input, output_shape, persistent_input=None):
        """Helper function for calculating an analytical ML model Jacobian.

        Args:
            model: tf.keras.Model instance.
            input: NumPy array of inputs to model about which the Jacobian should be computed.
            output_shape: tuple shape of model output, assumed to be correct.
            presistent_input: persistent tf.Variable for analytical Jacobian calculation.

        Returns:
            NumPy array of model Jacobian.
        """

        assert (
            persistent_input is not None
        ), "Must supply persistent tf.Variable to persistent_input for analytical Jacobian calculation."
        persistent_input.assign(np.expand_dims(input, axis=0))
        jacob_tf = self.calc_analytical_model_jacobian(model, persistent_input)
        jacob = tf.squeeze(jacob_tf, axis=[0, len(output_shape) + 1]).numpy()

        return jacob

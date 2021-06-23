import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from perform.rom.ml_library.ml_library import MLLibrary


class TFKerasFuncs(MLLibrary):
    """Class for accessing Tensorflow-Keras functionalities."""

    def __init__():

        super().__init__()

    def init_device(run_gpu):
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


    def load_model_obj(model_path):
        """Load Keras model object from file

        This function loads a trained Keras model from the older Keras H5 format.
        This does not accommodate the newer TensorFlow SavedModel format.

        Args:
            model_path: string path to *.h5 model file to be loaded.
        """

        model_obj = load_model(model_path, compile=False)
        return model_obj


    def get_io_shape(shape):
        """Gets Keras model I/O shape.

        This takes in the output of tf.keras.Model.Layer.input_shape or tf.keras.Model.Layer.output_shape.
        If the shape is already a tuple, simply returns the shape.
        If the shape is a list (as occasionally happens), it returns the shape tuple associated with this list.

        Args:
            shape: output of tf.keras.Model.Layer.input_shape or tf.keras.Model.Layer.output_shape
        """

        if type(shape) is list:
            if len(shape) != 1:
                raise ValueError("Invalid TF model I/O size")
            else:
                shape = shape[0]

        return shape
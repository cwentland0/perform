import numpy as np

from perform.input_funcs import catch_input
from perform.constants import FD_STEP_DEFAULT, REAL_TYPE


class MLLibrary:
    """Base class for abstracting machine learning library functionality.


    """

    def __init__(self, rom_domain):

        # initialize accelerator, if requested
        run_gpu = catch_input(rom_domain.rom_dict, "run_gpu", False)
        self.init_device(run_gpu)

    def check_model_io(self, model, input_shape, output_shape, isconv=False, io_format=None):
        """
        """

        if isconv:
            self.check_conv_io_format(io_format)
        self.check_io_shape(model, input_shape, output_shape)
        io_shapes = self.get_io_shape(model)
        io_dtypes = self.get_io_dtype(model)

        return (io_shapes, io_dtypes)

    def check_io_shape(self, model, input_shape, output_shape):
        """Check input/output dimensions and returns I/O dtypes

        Extracts shapes of the model's input and output laters and checks whether they match the expected shapes.

        Args:
            model:
            input_shape:
            output_shape:
        """

        input_shape_model, output_shape_model = self.get_io_shape(model)

        assert input_shape_model == input_shape, (
            "Mismatched model input shape: " + str(input_shape_model) + ", " + str(input_shape)
        )

        assert output_shape_model == output_shape, (
            "Mismatched model output shape: " + str(output_shape_model) + ", " + str(output_shape)
        )

    def get_shape_tuple(self, shape_var):
        """

        This takes in the output of tf.keras.Model.Layer.input_shape or tf.keras.Model.Layer.output_shape.
        If the shape is already a tuple, simply returns the shape.
        If the shape is a list (as occasionally happens), it returns the shape tuple associated with this list.
        """

        if type(shape_var) is list:
            if len(shape_var) != 1:
                raise ValueError("Invalid TF model I/O size")
            else:
                shape_var = shape_var[0]
        elif type(shape_var) is tuple:
            pass
        else:
            raise TypeError("Invalid shape input of type " + str(type(shape_var)))

        return shape_var

    def calc_numerical_model_jacobian(self, model, inputs, output_shape, fd_step=FD_STEP_DEFAULT):
        """Compute numerical Jacobian of model using finite-difference.

        Calculates the numerical Jacobian of a model with respect to the given inputs.
        A finite-difference approximation of the gradient with respect to each element of inputs is calculated.
        The fd_step attribute determines the finite difference step size.

        Args:
            model: model object for which the numerical Jacobian should be computed.
            inputs: NumPy array containing inputs about which the model Jacobian should be computed.

        Returns:
            NumPy array containing the numerical Jacobian.
        """

        jacob = np.zeros(output_shape + inputs.shape, dtype=REAL_TYPE)
        num_indices_output = np.product(output_shape)
        output_slice = ((np.s_[:],) * len(output_shape))

        # get initial prediction
        pred_base = self.infer_model(model, inputs)

        # get nd array indeces from linear indices
        num_indices_input = np.product(inputs.shape)
        pert_indices = list(map(tuple, np.stack(np.unravel_index(np.arange(num_indices_input), inputs.shape, order='C'), axis=1)))

        for elem_idx in range(num_indices_input):

            # perturb
            inputs_pert = inputs.copy()
            inputs_pert[pert_indices[elem_idx]] = inputs_pert[pert_indices[elem_idx]] + fd_step

            # make prediction at perturbed state
            pred = self.infer_model(model, inputs_pert)

            # compute finite difference approximation
            jacob_slice = output_slice + pert_indices[elem_idx]
            jacob[jacob_slice] = (pred - pred_base) / fd_step

        return jacob

import numpy as np

from perform.input_funcs import catch_input
from perform.constants import FD_STEP_DEFAULT, REAL_TYPE


class MLLibrary:
    """Base class for abstracting machine learning library functionality.

    Child classes implement library-specific (e.g. Tensorflow, PyTorch) routines for accomplishing basic
    machine learning tasks, such as model inference or device memory management.
    This class implements higher-level tasks which utilize various child class routines.
    
    Routines here and in child classes should be written to be intentionally general and leave most specifics
    to be implemented in RomModel child classes.

    Args:
        rom_domain: RomDomain which contains relevant ROM input dictionary.
    """

    def __init__(self, rom_domain):

        # initialize accelerator, if requested
        run_gpu = catch_input(rom_domain.rom_dict, "run_gpu", False)
        self.init_device(run_gpu)

    def check_model_io(self, model, input_shape, output_shape, isconv=False, io_format=None):
        """Checks ML model hardware compatabilty and I/O shape against expected shapes.

        Args:
            model: ML model object.
            input_shape: tuple of expected model input shape.
            output_shape: tuple of expected model output shape.
            isconv: 
                if True, indicates that model contains convolutional input or output layers,
                and checks whether hardware running ML model supports the format given by io_format.
            io_format: 
                string indicating convolutional input/output format,
                either "channels_first" (i.e. NCHW) or "channels_last" (i.e. NHWC)

        Returns:
            List of model I/O tuple shapes and list of model I/O data types.
        """

        if isconv:
            self.check_conv_io_format(io_format)
        self.check_io_shape(model, input_shape, output_shape)
        io_shapes = self.get_io_shape(model)
        io_dtypes = self.get_io_dtype(model)

        return (io_shapes, io_dtypes)

    def check_io_shape(self, model, input_shape, output_shape):
        """Check input/output dimensions against expected shapes.

        Args:
            model: ML model object.
            input_shape: tuple of expected model input shape.
            output_shape: tuple of expected model output shape.
        """

        input_shape_model, output_shape_model = self.get_io_shape(model)

        assert input_shape_model == input_shape, (
            "Mismatched model input shape: " + str(input_shape_model) + ", " + str(input_shape)
        )

        assert output_shape_model == output_shape, (
            "Mismatched model output shape: " + str(output_shape_model) + ", " + str(output_shape)
        )

    def get_shape_tuple(self, shape_var):
        """Get shape tuple from layer I/O shape.

        Takes in shape of model layer (usually input of first layer or output of last layer).
        If the shape is already a tuple, simply returns the shape.
        If the shape is a list (as may happen), it returns the shape tuple associated with this list.

        Args:
            shape_var: either a tuple shape or single-element list containing a tuple shape.
        """

        if type(shape_var) is list:
            if len(shape_var) != 1:
                raise ValueError("Invalid model I/O size")
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
        A finite-difference approximation of the gradient with respect to each element of the inputs is calculated.
        The fd_step attribute determines the finite difference step size.
        
        This method is agnostic to the number of input/output dimensions and their shapes, but assumes that
        output_shape is correct.

        Args:
            model: model object for which the numerical Jacobian is to be computed.
            inputs: NumPy array containing inputs about which the model Jacobian should be computed.
            output_shape: tuple shape of model output.
            fd_step: float size of finite difference step.

        Returns:
            NumPy array containing the numerical Jacobian.
        """

        # prep Jacobian and indices into Jacobian 
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

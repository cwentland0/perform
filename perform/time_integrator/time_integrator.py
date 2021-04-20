class TimeIntegrator:
    """Base class for all numerical time integrators.

    Child classes implement explicit and implicit time integration schemes.

    Each SolutionDomain has its own TimeIntegrator, which permits ROM-ROM and FOM-ROM coupling
    with various time integrators.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        dt: Physical time step size, in seconds.
        time_scheme:
            String name of the numerical time integration scheme to apply to the SolutionDomain with which
            this TimeIntegrator is associated.
        time_order:
            Time order of accuracy. Is fixed for some time integrators, and a limited range is supported by others.
        subiter: Zero-indexed physical time step subiteration number.
    """

    def __init__(self, param_dict):

        self.dt = float(param_dict["dt"])
        self.time_scheme = str(param_dict["time_scheme"])
        self.time_order = int(param_dict["time_order"])
        assert self.time_order >= 1, "time_order only accepts positive integer values."

        self.subiter = 0

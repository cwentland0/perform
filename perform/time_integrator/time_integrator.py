

class TimeIntegrator:
    """
    Base class for time integrators
    """

    def __init__(self, param_dict):

        self.dt = float(param_dict["dt"])
        self.time_scheme = str(param_dict["time_scheme"])
        self.time_order = int(param_dict["time_order"])
        assert (self.time_order >= 1), (
            "time_order only accepts positive integer values.")

        self.subiter = 0

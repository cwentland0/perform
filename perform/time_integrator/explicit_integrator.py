import time

import numpy as np

from perform.constants import REAL_TYPE
from perform.time_integrator.time_integrator import TimeIntegrator


class ExplicitIntegrator(TimeIntegrator):
    """Base class for explicit time integrators.

    All child classes must implement the solve_sol_change() member function.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        time_type: Set to "explicit".
        dual_time: Set to False.
        adapt_dtau: Set to False.
    """

    def __init__(self, param_dict):

        super().__init__(param_dict)

        self.time_type = "explicit"
        self.dual_time = False
        self.adapt_dtau = False


# ----- Runge-Kutta integrators -----


class RKExplicit(ExplicitIntegrator):
    """Base class for explicit Runge-Kutta time integrators:

    All child classes must simply define their Butcher tableu via the attributes rk_a, rk_b, and rk_c.
    Additionally, each must set the variable subiter_max.

    Please refer to the solver theory documentation for more details on each method.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        rk_rhs: List of NumPy arrays of prior RK iteration RHS evaluations.
    """

    def __init__(self, param_dict):

        super().__init__(param_dict)

        self.rk_rhs = [None] * self.subiter_max  # subiteration RHS history

    def solve_sol_change(self, rhs):
        """Compute intermediate step or final physical time step.

        Args:
            rhs: Semi-discrete governing ODE right-hand side function evaluation.

        Returns:
            NumPy array of the change in the solution profile.
        """

        self.rk_rhs[self.subiter] = rhs.copy()

        if self.subiter == (self.subiter_max - 1):
            dsol = self.solve_sol_change_iter(rhs)
        else:
            dsol = self.solve_sol_change_subiter(rhs)

        dsol *= self.dt

        return dsol

    def solve_sol_change_subiter(self, rhs):
        """Change in intermediate solution for subiteration.

        Args:
            rhs: NumPy array of the semi-discrete governing ODE right-hand side function evaluation.

        Returns:
            NumPy array of the change in the solution profile between subiterations.
        """

        dsol = np.zeros(rhs.shape, dtype=REAL_TYPE)
        for rk_iter in range(self.subiter + 1):
            rk_a = self.rk_a_vals[self.subiter + 1, rk_iter]
            if rk_a != 0.0:
                dsol += rk_a * self.rk_rhs[rk_iter]

        return dsol

    def solve_sol_change_iter(self, rhs):
        """Compute change in physical solution at final subiteration.

        Args:
            rhs: NumPy array of the semi-discrete governing ODE right-hand side function evaluation.

        Returns:
            NumPy array of the change in the solution profile for final subiteration.
        """

        dsol = np.zeros(rhs.shape, dtype=REAL_TYPE)
        for rk_iter in range(self.subiter_max):
            rk_b = self.rk_b_vals[rk_iter]
            if rk_b != 0.0:
                dsol += rk_b * self.rk_rhs[rk_iter]

        return dsol


class ClassicRK4(RKExplicit):
    """Classic explicit RK4 scheme.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        subiter_max: Total number of subiterations for one physical time step.
    """

    def __init__(self, param_dict):

        self.subiter_max = 4

        super().__init__(param_dict)

        if self.time_order != 4:
            print("classic_rk4 is fourth-order accurate, but you set time_order = " + str(self.time_order))
            print("Continuing, set time_order = 4 to get rid of this warning")
            time.sleep(0.5)

        self.rk_a_vals = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        self.rk_b_vals = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
        self.rk_c_vals = np.array([0.0, 0.5, 0.5, 1.0])


class SSPRK3(RKExplicit):
    """Strong stability-preserving explicit RK3 scheme.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        subiter_max: Total number of subiterations for one physical time step.
    """

    def __init__(self, param_dict):

        self.subiter_max = 3

        super().__init__(param_dict)

        if self.time_order != 3:
            print("ssp_rk3 is third-order accurate, but you set time_order = " + str(self.time_order))
            print("Continuing, set time_order = 3 to get rid of this warning")
            time.sleep(0.5)

        self.rk_a_vals = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        self.rk_b_vals = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        self.rk_c_vals = np.array([0.0, 1.0, 0.5])


class JamesonLowStore(RKExplicit):
    """ "Low-storage" class of RK schemes by Jameson.

    Not actually appropriate for unsteady problems, supposedly.
    Not actually low-storage to work with general RK format, just maintained here for consistency with old code.

    Args:
        param_dict: Dictionary of parameters read from the solver parameters input file.

    Attributes:
        subiter_max: Total number of subiterations for one physical time step.
    """

    def __init__(self, param_dict):

        # have to get this early to set subiter_max
        time_order = int(param_dict["time_order"])
        self.subiter_max = time_order

        super().__init__(param_dict)

        # if you add another row and column to rk_a_vals, update this assertion
        assert time_order <= 4, "Jameson low-storage RK scheme for order " + str(time_order) + " not yet implemented"

        self.rk_a_vals = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]]
        )

        self.rk_a_vals = self.rk_a_vals[-time_order:, -time_order:]

        self.rk_b_vals = np.zeros(time_order, dtype=REAL_TYPE)
        self.rk_b_vals[-1] = 1.0
        self.rk_c_vals = np.zeros(time_order, dtype=REAL_TYPE)


# ----- End Runge-Kutta integrators -----

# TODO: Add "integrator" for ROM models that just
# 	update from one state to the next,
# 	presumably calls some generic update method.
# 	This way, can still have a code history for
# 	methods that need the history but don't
# 	need numerical time integration (e.g. TCN)

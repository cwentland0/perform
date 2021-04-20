import numpy as np


class Limiter:
    """Base class for gradient limiters.

    Simply provides some member functions which may be common to multiple child class limiters.

    All child classes must implement the member method calc_limiter().
    """

    def __init__(self):

        pass

    def calc_neighbor_minmax(self, sol):
        """Find minimum and maximum of cell state and neighbor cell states

        Args:
            sol:
                2D NumPy array of a solution profile (e.g. SolutionPhys.sol_prim),
                where the first axis are the solution variables and the second axis are the spatial locations.

        Returns:
            NumPy arrays of the neighbor minimums and neighbor maximums, both of the same shape as sol.
        """

        # max and min of cell and neighbors
        sol_max = sol.copy()
        sol_min = sol.copy()

        # first compare against right neighbor
        sol_max[:, :-1] = np.maximum(sol[:, :-1], sol[:, 1:])
        sol_min[:, :-1] = np.minimum(sol[:, :-1], sol[:, 1:])

        # then compare agains left neighbor
        sol_max[:, 1:] = np.maximum(sol_max[:, 1:], sol[:, :-1])
        sol_min[:, 1:] = np.minimum(sol_min[:, 1:], sol[:, :-1])

        return sol_min, sol_max

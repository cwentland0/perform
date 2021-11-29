import numpy as np

from perform.constants import REAL_TYPE
from perform.limiter.limiter import Limiter


class BarthJespLimiter(Limiter):
    """Baseline Barth-Jespersen limiter.

    Child classes implement variations of the limiter of Barth and Jespersen (1989) in one dimension.
    Ensures that no new minima or maxima are introduced in reconstruction, but is non-differentiable.
    """

    def __init__(self):

        super().__init__()

    def calc_limiter(self, sol_domain, sol_full, grad):
        """Compute multiplicative limiter.

        Args:
            sol_full:
            sol_domain: SolutionDomain with which this Limiter is associated.
            grad: NumPy array of the un-limited gradient directly computed from finite difference stencil.

        Returns:
            NumPy array of the multiplicative gradient limiter profile.
        """

        sol = sol_full[:, sol_domain.grad_idxs]

        # get min/max of cell and neighbors
        sol_min, sol_max = self.calc_neighbor_minmax(sol_full[:, sol_domain.grad_neigh_idxs])

        # extract gradient cells
        sol_min = sol_min[:, sol_domain.grad_neigh_extract]
        sol_max = sol_max[:, sol_domain.grad_neigh_extract]

        # unconstrained reconstruction at neighboring cell centers
        d_sol = self.calc_d_sol(grad, sol_domain.mesh.dx)
        sol_left = sol - d_sol
        sol_right = sol + d_sol

        # limiter defaults to 1
        phi_left = np.ones(sol.shape, dtype=REAL_TYPE)
        phi_right = np.ones(sol.shape, dtype=REAL_TYPE)

        # find idxs where difference is either positive or negative
        cond1_left = (sol_left - sol) > 0
        cond1_right = (sol_right - sol) > 0
        cond2_left = (sol_left - sol) < 0
        cond2_right = (sol_right - sol) < 0

        # threshold limiter for left and right reconstruction
        phi_left[cond1_left] = np.minimum(
            1.0, (sol_max[cond1_left] - sol[cond1_left]) / (sol_left[cond1_left] - sol[cond1_left])
        )
        phi_right[cond1_right] = np.minimum(
            1.0, (sol_max[cond1_right] - sol[cond1_right]) / (sol_right[cond1_right] - sol[cond1_right]),
        )
        phi_left[cond2_left] = np.minimum(
            1.0, (sol_min[cond2_left] - sol[cond2_left]) / (sol_left[cond2_left] - sol[cond2_left])
        )
        phi_right[cond2_right] = np.minimum(
            1.0, (sol_min[cond2_right] - sol[cond2_right]) / (sol_right[cond2_right] - sol[cond2_right]),
        )

        # take minimum limiter from left and right
        phi = np.minimum(phi_left, phi_right)

        return phi


class BarthJespCellLimiter(BarthJespLimiter):
    """Barth-Jespersen limiter, based on reconstruction at cell centers."""

    def __init__(self):

        super().__init__()

    def calc_d_sol(self, grad, dx):

        return grad * dx


class BarthJespFaceLimiter(BarthJespLimiter):
    """Barth-Jespersen limiter, based on reconstruction at face instead of cell centers."""

    def __init__(self):

        super().__init__()

    def calc_d_sol(self, grad, dx):

        return grad * dx / 2.0

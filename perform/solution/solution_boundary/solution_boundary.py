from math import sin, pi

import numpy as np

from perform.constants import REAL_TYPE
from perform.solution.solution_phys import SolutionPhys
from perform.input_funcs import parse_bc


class SolutionBoundary(SolutionPhys):
    """Physical solution for boundary ghost cells.

    This class provides a base for SolutionInlet and SolutionOutlet, reading input parameters,
    initializing boundary state calculation functions, and executing those functions during the simulation runtime.

    Each SolutionDomain includes two SolutionBoundary's: one SolutionInlet and one SolutionOutlet.

    The interpretations of the attributes press, vel, temp, mass_fracs, and rho are generally ambiguous and rarely
    represent such fixed values at the boundary. Please refer to the documentation for how to interpret each of these
    values for a given boundary condition.

    Args:
        gas: GasModel associated with the SolutionDomain with which this SolutionPhys is associated.
        solver: SystemSolver containing global simulation parameters.
        bound_type: Either "inlet" or "outlet", for an inlet or outlet boundary ghost cell, respectively.

    Attributes:
        press: Pressure-related quantity, interpreted depending on boundary condition.
        vel: Velocity-related quantity, interpreted depending on boundary condition.
        temp: Temperature-related quantity, interpreted depending on boundary condition.
        mass_fracs: Mass fraction-related quantity, interpreted depending on boundary condition.
        rho: Density-related quantity, interpreted depending on boundary condition.
        pert_type:
            String name of the type of boundary perturbation to apply, interpreted depending on boundary condition.
        pert_perc:
            Percentage of the fixed boundary value about which to apply the boundary perturbation.
            For example, a 10% perturbation would be supplied as 0.1.
        pert_freq: List of superimposed frequencies at which to apply the boundary perturbation, in Hz
    """

    def __init__(self, gas, solver, bound_type):

        param_dict = solver.param_dict

        # Read input boundary values
        (
            self.press,
            self.vel,
            self.temp,
            self.mass_fracs,
            self.rho,
            self.pert_type,
            self.pert_perc,
            self.pert_freq,
        ) = parse_bc(bound_type, param_dict)

        assert (
            len(self.mass_fracs) == gas.num_species_full
        ), "Must provide mass fraction state for all species at boundary"
        assert np.sum(self.mass_fracs) == 1.0, "Boundary mass fractions must sum to 1.0"

        # This will be updated at each iteration, just initializing with a reasonable number
        sol_dummy = np.zeros((gas.num_eqs, 1), dtype=REAL_TYPE)
        sol_dummy[0, 0] = 1e6
        sol_dummy[1, 0] = 1.0
        sol_dummy[2, 0] = 300.0
        sol_dummy[3, 0] = 1.0
        super().__init__(gas, 1, sol_prim_in=sol_dummy)
        self.sol_prim[3:, 0] = self.mass_fracs[gas.mass_frac_slice]

    def calc_pert(self, sol_time):
        """Compute sinusoidal perturbation factor.

        Args:
            sol_time: Current physical time, in seconds.
        """

        # TODO: add phase offset

        pert = 0.0
        for f in self.pert_freq:
            pert += sin(2.0 * pi * self.pert_freq * sol_time)
        pert *= self.pert_perc

        return pert

    def calc_boundary_state(self, sol_time, space_order, sol_prim=None, sol_cons=None):
        """Run boundary calculation and update ghost cell state.

        Assumed that boundary function sets primitive state.

        Args:
            sol_time: Current physical time, in seconds.
            space_order: Spatial order of accuracy of face reconstruction.
            sol_prim: NumPy array of SolutionInterior.sol_prim, the primitive state profile.
            sol_cons: NumPy array of SolutionInterior.sol_cons, the conservative state profile.
        """

        self.bound_func(sol_time, space_order, sol_prim=sol_prim, sol_cons=sol_cons)

        self.update_state(from_cons=False)

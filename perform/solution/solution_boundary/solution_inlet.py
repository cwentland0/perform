from math import pow, sqrt

from perform.solution.solution_boundary.solution_boundary import SolutionBoundary


class SolutionInlet(SolutionBoundary):
    """
    Inlet ghost cell solution
    """

    def __init__(self, gas, solver):

        param_dict = solver.param_dict
        self.bound_cond = param_dict["bound_cond_inlet"]

        # Add assertions to check that required properties are specified
        if self.bound_cond == "stagnation":
            self.bound_func = self.calc_stagnation_bc
        elif self.bound_cond == "fullstate":
            self.bound_func = self.calc_full_state_bc
        elif self.bound_cond == "meanflow":
            self.bound_func = self.calc_mean_flow_bc
        else:
            raise ValueError("Invalid inlet boundary condition selection: " + str(self.bound_cond))

        super().__init__(gas, solver, "inlet")

    def calc_stagnation_bc(self, sol_time, space_order, sol_prim=None, sol_cons=None):
        """
        Specify stagnation temperature and stagnation pressure
        """

        assert sol_prim is not None, "Must provide primitive interior state"

        # chemical composition assumed constant near boundary
        r_mix = self.r_mix[0]
        gamma_mix = self.gamma_mix[0]
        gamma_mix_m1 = gamma_mix - 1.0

        # interior state
        vel_p1 = sol_prim[1, 0]
        vel_p2 = sol_prim[1, 1]
        c_p1 = sqrt(gamma_mix * r_mix * sol_prim[2, 0])
        c_p2 = sqrt(gamma_mix * r_mix * sol_prim[2, 1])

        # Interpolate outgoing Riemann invariant
        # Negative sign on velocity is to account
        # for flux/boundary normal directions
        j1 = -vel_p1 - (2.0 * c_p1) / gamma_mix_m1
        j2 = -vel_p2 - (2.0 * c_p2) / gamma_mix_m1

        # Extrapolate to exterior
        if space_order == 1:
            J = j1
        elif space_order == 2:
            J = 2.0 * j1 - j2
        else:
            raise ValueError(
                "Higher order extrapolation implementation " + "required for spatial order " + str(space_order)
            )

        # Quadratic form for exterior Mach number
        c2 = gamma_mix * r_mix * self.temp

        a_val = c2 - J ** 2 * gamma_mix_m1 / 2.0
        b_val = (4.0 * c2) / gamma_mix_m1
        c_val = (4.0 * c2) / gamma_mix_m1 ** 2 - J ** 2
        rad = b_val ** 2 - 4.0 * a_val * c_val

        # Check for non-physical solution (usually caused by reverse flow)
        if rad < 0.0:
            print("a_val: " + str(a_val))
            print("b_val: " + str(b_val))
            print("c_val: " + str(c_val))
            print("Boundary velocity: " + str(vel_p1))
            raise ValueError("Non-physical inlet state")

        # Solve quadratic formula, assign Mach number
        # depending on sign/magnitude
        # If only one positive, select that.
        # If both positive, select smaller
        rad = sqrt(rad)
        mach_1 = (-b_val - rad) / (2.0 * a_val)
        mach_2 = (-b_val + rad) / (2.0 * a_val)
        if (mach_1 > 0) and (mach_2 > 0):
            mach_bound = min(mach_1, mach_2)
        elif (mach_1 <= 0) and (mach_2 <= 0):
            raise ValueError("Non-physical Mach number at inlet")
        else:
            mach_bound = max(mach_1, mach_2)

        # Compute exterior state
        temp_bound = self.temp / (1.0 + gamma_mix_m1 / 2.0 * mach_bound ** 2)
        self.sol_prim[2, 0] = temp_bound
        self.sol_prim[0, 0] = self.press * pow(temp_bound / self.temp, gamma_mix / gamma_mix_m1)
        c_bound = sqrt(gamma_mix * r_mix * temp_bound)
        self.sol_prim[1, 0] = mach_bound * c_bound

    def calc_full_state_bc(self, sol_time, space_order, sol_prim=None, sol_cons=None):
        """
        Full state specification
        Mostly just for perturbing inlet state to check for outlet reflections
        """

        press_bound = self.press
        vel_bound = self.vel
        temp_bound = self.temp

        # Perturbation
        if self.pert_type == "pressure":
            press_bound *= 1.0 + self.calc_pert(sol_time)
        elif self.pert_type == "velocity":
            vel_bound *= 1.0 + self.calc_pert(sol_time)
        elif self.pert_type == "temperature":
            press_bound *= 1.0 + self.calc_pert(sol_time)

        # Compute ghost cell state
        self.sol_prim[0, 0] = press_bound
        self.sol_prim[1, 0] = vel_bound
        self.sol_prim[2, 0] = temp_bound

    def calc_mean_flow_bc(self, sol_time, space_order, sol_prim=None, sol_cons=None):
        """
        Non-reflective boundary, unsteady solution
        is perturbation about mean flow solution
        Refer to documentation for derivation
        """

        assert sol_prim is not None, "Must provide primitive interior state"

        # Mean flow and infinitely-far upstream quantities
        press_up = self.press
        temp_up = self.temp
        mass_fracs_up = self.mass_fracs[:-1]
        rho_c_mean = self.vel
        rho_cp_mean = self.rho

        if self.pert_type == "pressure":
            press_up *= 1.0 + self.calc_pert(sol_time)

        # Interior quantities
        press_in = sol_prim[0, :2]
        vel_in = sol_prim[1, :2]

        # Characteristic variables
        w_3_in = vel_in - press_in / rho_c_mean

        # Extrapolate to exterior
        if space_order == 1:
            w_3_bound = w_3_in[0]
        elif space_order == 2:
            w_3_bound = 2.0 * w_3_in[0] - w_3_in[1]
        else:
            raise ValueError(
                "Higher order extrapolation implementation " + "required for spatial order " + str(space_order)
            )

        # Compute exterior state
        press_bound = (press_up - w_3_bound * rho_c_mean) / 2.0
        self.sol_prim[0, 0] = press_bound
        self.sol_prim[1, 0] = (press_up - press_bound) / rho_c_mean
        self.sol_prim[2, 0] = temp_up + (press_bound - press_up) / rho_cp_mean

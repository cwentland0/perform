import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.invisc_flux.invisc_flux import InviscFlux


class RoeInviscFlux(InviscFlux):
    """Class implementing flux methods for Roe's flux difference scheme.

    Inherits from InviscFlux. Provides member functions for computing the numerical flux at each face and
    its Jacobian with respect to the primitive and conservative state.

    Please refer to Roe (1981) for details on the Roe scheme.

    Args:
        sol_domain: SolutionDomain with which this Flux is associated.
    """

    def __init__(self, sol_domain):

        super().__init__()

    def calc_avg_state(self, sol_left, sol_right, sol_ave):
        """Computes average state at cell faces.

        Computes the special Roe average, first by computing the Roe average density and stagnation enthalpy, then
        adjusting the primitive state iteratively to be consistent.

        Args:
            sol_left: SolutionPhys representing the solution on the left side of cell faces.
            sol_right: SolutionPhys representing the solution on the right side of cell faces.
            sol_ave: SolutionPhys where the face average state will be stored.
        """

        # Useful factors
        sqrhol = np.sqrt(sol_left.sol_cons[0, :])
        sqrhor = np.sqrt(sol_right.sol_cons[0, :])
        fac = sqrhol / (sqrhol + sqrhor)
        fac1 = 1.0 - fac

        # Roe average stagnation enthalpy and density
        sol_ave.h0 = fac * sol_left.h0 + fac1 * sol_right.h0
        sol_ave.sol_cons[0, :] = sqrhol * sqrhor

        # First guess at Roe average primitive state
        sol_ave.sol_prim = fac[None, :] * sol_left.sol_prim + fac1[None, :] * sol_right.sol_prim
        sol_ave.mass_fracs_full = sol_ave.gas_model.calc_all_mass_fracs(sol_ave.sol_prim[3:, :], threshold=True)

        # Adjust primitive state iteratively to conform to Roe average density and enthalpy, update state
        sol_ave.calc_state_from_rho_h0()

    def calc_flux(self, sol_domain):
        """Compute numerical inviscid flux vector.

        Args:
            sol_domain: SolutionDomain with which this Flux is associated.

        Returns:
            NumPy array of numerical inviscid flux for every governing equation at each finite volume face.
        """

        # TODO: entropy fix

        sol_left = sol_domain.sol_left
        sol_right = sol_domain.sol_right

        # Compute inviscid flux vectors of left and right state
        flux_left = self.calc_inv_flux(sol_left.sol_cons, sol_left.sol_prim, sol_left.h0)
        flux_right = self.calc_inv_flux(sol_right.sol_cons, sol_right.sol_prim, sol_right.h0)

        # Dissipation term
        d_sol_prim = sol_left.sol_prim - sol_right.sol_prim
        sol_domain.roe_diss = self.calc_roe_diss(sol_domain.sol_ave)
        diss_term = 0.5 * (sol_domain.roe_diss * np.expand_dims(d_sol_prim, 0)).sum(-2)

        # Complete Roe flux
        flux = 0.5 * (flux_left + flux_right) + diss_term

        return flux

    def calc_roe_diss(self, sol_ave):
        """Compute dissipation term of Roe flux.

        The derivation of this term is provided in the solver theory documentation.

        Args:
            sol_ave: SolutionPhys of the Roe average state at each finite volume face.

        Returns:
            3D NumPy array of the Roe dissipation matrix.
        """

        gas = sol_ave.gas_model

        diss_matrix = np.zeros((gas.num_eqs, gas.num_eqs, sol_ave.num_cells), dtype=REAL_TYPE)

        # For clarity
        rho = sol_ave.sol_cons[0, :]
        vel = sol_ave.sol_prim[1, :]
        mass_fracs = sol_ave.sol_prim[3:, :]
        h0 = sol_ave.h0
        c = sol_ave.c

        # Derivatives of density and enthalpy
        sol_ave.update_density_enthalpy_derivs()
        d_rho_d_press = sol_ave.d_rho_d_press
        d_rho_d_temp = sol_ave.d_rho_d_temp
        d_rho_d_mass_frac = sol_ave.d_rho_d_mass_frac
        d_enth_d_press = sol_ave.d_enth_d_press
        d_enth_d_temp = sol_ave.d_enth_d_temp
        d_enth_d_mass_frac = sol_ave.d_enth_d_mass_frac

        # Gamma terms for energy equation
        g_press = rho * d_enth_d_press + d_rho_d_press * h0 - 1.0
        g_temp = rho * d_enth_d_temp + d_rho_d_temp * h0
        g_mass_frac = rho[None, :] * d_enth_d_mass_frac + h0[None, :] * d_rho_d_mass_frac

        # Characteristic speeds
        lambda1 = vel + c
        lambda2 = vel - c
        lambda1_abs = np.absolute(lambda1)
        lambda2_abs = np.absolute(lambda2)

        r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
        alpha = c * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
        beta = np.power(c, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
        phi = c * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

        eta = (1.0 - rho * d_enth_d_press) / d_enth_d_temp
        psi = eta * d_rho_d_temp + rho * d_rho_d_press

        vel_abs = np.absolute(vel)

        beta_star = beta * psi
        beta_e = beta * (rho * g_press + g_temp * eta)
        phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rho
        phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rho
        m = rho * alpha
        e = rho * vel * alpha

        # Continuity equation row
        diss_matrix[0, 0, :] = phi_star
        diss_matrix[0, 1, :] = beta_star
        diss_matrix[0, 2, :] = vel_abs * d_rho_d_temp
        diss_matrix[0, 3:, :] = vel_abs[None, :] * d_rho_d_mass_frac

        # Momentum equation row
        diss_matrix[1, 0, :] = vel * phi_star + r_roe
        diss_matrix[1, 1, :] = vel * beta_star + m
        diss_matrix[1, 2, :] = vel * vel_abs * d_rho_d_temp
        diss_matrix[1, 3:, :] = (vel * vel_abs)[None, :] * d_rho_d_mass_frac

        # Energy equation row
        diss_matrix[2, 0, :] = phi_e + r_roe * vel
        diss_matrix[2, 1, :] = beta_e + e
        diss_matrix[2, 2, :] = g_temp * vel_abs
        diss_matrix[2, 3:, :] = g_mass_frac * vel_abs[None, :]

        # Species transport row
        diss_matrix[3:, 0, :] = mass_fracs * phi_star[None, :]
        diss_matrix[3:, 1, :] = mass_fracs * beta_star[None, :]
        diss_matrix[3:, 2, :] = mass_fracs * (vel_abs * d_rho_d_temp)[None, :]
        # TODO: vectorize
        for mf_idx_out in range(3, gas.num_eqs):
            for mf_idx_in in range(3, gas.num_eqs):
                # TODO: check this again against GEMS,
                # something weird going on
                if mf_idx_out == mf_idx_in:
                    diss_matrix[mf_idx_out, mf_idx_in, :] = vel_abs * (
                        rho + mass_fracs[mf_idx_out - 3, :] * d_rho_d_mass_frac[mf_idx_in - 3, :]
                    )
                else:
                    diss_matrix[mf_idx_out, mf_idx_in, :] = (
                        vel_abs * mass_fracs[mf_idx_out - 3, :] * d_rho_d_mass_frac[mf_idx_in - 3, :]
                    )

        return diss_matrix

    def calc_jacob(self, sol_domain, wrt_prim):
        """Compute numerical inviscid flux Jacobian.

        Calculates flux Jacobian at each face and assembles Jacobian with respect to each
        finite volume cell's state. Note that the gradient with respect to boundary ghost cell states are
        excluded, as the Newton iteration linear solve does not need this.

        Args:
            sol_domain: SolutionDomain with which this Flux is associated.
            wrt_prim:
                Boolean flag. If True, calculate Jacobian w/r/t the primitive variables.
                If False, calculate w/r/t conservative variables.

        Returns:
            jacob_center_cell: center block diagonal of flux Jacobian, representing the gradient of a given cell's
            viscous flux contribution with respect to its own state.
            jacob_left_cell: lower block diagonal of flux Jacobian, representing the gradient of a given cell's
            viscous flux contribution with respect to its left neighbor's state.
            jacob_left_cell: upper block diagonal of flux Jacobian, representing the gradient of a given cell's
            viscous flux contribution with respect to its right neighbor's state.
        """

        roe_diss = sol_domain.roe_diss
        center_samp = sol_domain.flux_rhs_idxs
        left_samp = sol_domain.jacob_left_samp
        right_samp = sol_domain.jacob_right_samp

        # Jacobian of inviscid flux vector at left and right face reconstruction
        jacob_face_left = self.calc_d_inv_flux_d_sol_prim(sol_domain.sol_left)
        jacob_face_right = self.calc_d_inv_flux_d_sol_prim(sol_domain.sol_right)
        # TODO: when conservative Jacobian is implemented, uncomment this
        # if wrt_prim:
        #     jacob_face_left = self.calc_d_inv_flux_d_sol_prim(sol_domain.sol_left)
        #     jacob_face_right = self.calc_d_inv_flux_d_sol_prim(sol_domain.sol_right)
        # else:
        #     raise ValueError("Roe disspation issue not addressed yet")
        #     jacob_face_left = self.calc_d_inv_flux_d_sol_cons(sol_domain.sol_left)
        #     jacob_face_right = self.calc_d_inv_flux_d_sol_cons(sol_domain.sol_right)

        # center, lower, and upper block diagonal of full numerical flux Jacobian
        jacob_center_cell = (jacob_face_left[:, :, center_samp + 1] + roe_diss[:, :, center_samp + 1]) + (
            -jacob_face_right[:, :, center_samp] + roe_diss[:, :, center_samp]
        )
        jacob_left_cell = -jacob_face_left[:, :, left_samp] - roe_diss[:, :, left_samp]
        jacob_right_cell = jacob_face_right[:, :, right_samp] - roe_diss[:, :, right_samp]

        jacob_center_cell *= 0.5
        jacob_left_cell *= 0.5
        jacob_right_cell *= 0.5

        return jacob_center_cell, jacob_left_cell, jacob_right_cell

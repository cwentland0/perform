import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.invisc_flux.invisc_flux import InviscFlux


class RoeInviscFlux(InviscFlux):
    """
    Roe's flux difference scheme
    """

    def __init__(self, sol_domain):

        super().__init__()

    def calc_flux(self, sol_domain):
        """
        Compute flux array
        """

        # TODO: entropy fix

        sol_left = sol_domain.sol_left
        sol_right = sol_domain.sol_right
        sol_prim_left = sol_left.sol_prim
        sol_cons_left = sol_left.sol_cons
        sol_prim_right = sol_right.sol_prim
        sol_cons_right = sol_domain.sol_right.sol_cons
        gas_model = sol_domain.gas_model

        # Inviscid flux vector
        flux_left = np.zeros(sol_prim_left.shape, dtype=REAL_TYPE)
        flux_right = np.zeros(sol_prim_right.shape, dtype=REAL_TYPE)

        # Compute sqrhol, sqrhor, fac, and fac1
        sqrhol = np.sqrt(sol_cons_left[0, :])
        sqrhor = np.sqrt(sol_cons_right[0, :])
        fac = sqrhol / (sqrhol + sqrhor)
        fac1 = 1.0 - fac

        # Roe average stagnation enthalpy and density
        sol_left.hi = gas_model.calc_spec_enth(sol_prim_left[2, :])
        sol_left.h0 = gas_model.calc_stag_enth(sol_prim_left[1, :], sol_left.mass_fracs_full, spec_enth=sol_left.hi)
        sol_right.hi = gas_model.calc_spec_enth(sol_prim_right[2, :])
        sol_right.h0 = gas_model.calc_stag_enth(sol_prim_right[1, :], sol_right.mass_fracs_full, spec_enth=sol_right.hi)

        sol_ave = sol_domain.sol_ave
        sol_ave.h0 = fac * sol_left.h0 + fac1 * sol_right.h0
        sol_ave.sol_cons[0, :] = sqrhol * sqrhor

        # Compute Roe average primitive state
        sol_ave.sol_prim = fac[None, :] * sol_prim_left + fac1[None, :] * sol_prim_right
        sol_ave.mass_fracs_full = gas_model.calc_all_mass_fracs(sol_ave.sol_prim[3:, :], threshold=True)

        # Adjust iteratively to conform to Roe average density and enthalpy
        sol_ave.calc_state_from_rho_h0()

        # Compute Roe average state at faces, associated fluid properties
        sol_ave.calc_state_from_prim(calc_r=True, calc_cp=True)
        sol_ave.gamma_mix = gas_model.calc_mix_gamma(sol_ave.r_mix, sol_ave.cp_mix)
        sol_ave.c = gas_model.calc_sound_speed(
            sol_ave.sol_prim[2, :],
            r_mix=sol_ave.r_mix,
            gamma_mix=sol_ave.gamma_mix,
            mass_fracs=sol_ave.sol_prim[3:, :],
            cp_mix=sol_ave.cp_mix,
        )

        # Compute inviscid flux vectors of left and right state
        flux_left[0, :] = sol_cons_left[1, :]
        flux_left[1, :] = sol_cons_left[1, :] * sol_prim_left[1, :] + sol_prim_left[0, :]
        flux_left[2, :] = sol_cons_left[0, :] * sol_left.h0 * sol_prim_left[1, :]
        flux_left[3:, :] = sol_cons_left[3:, :] * sol_prim_left[[1], :]
        flux_right[0, :] = sol_cons_right[1, :]
        flux_right[1, :] = sol_cons_right[1, :] * sol_prim_right[1, :] + sol_prim_right[0, :]
        flux_right[2, :] = sol_cons_right[0, :] * sol_right.h0 * sol_prim_right[1, :]
        flux_right[3:, :] = sol_cons_right[3:, :] * sol_prim_right[[1], :]

        # Maximum wave speed for adapting dtau, if needed
        # TODO: need to adaptively size this for hyper-reduction
        if sol_domain.time_integrator.adapt_dtau:
            srf = np.maximum(sol_ave.sol_prim[1, :] + sol_ave.c, sol_ave.sol_prim[1, :] - sol_ave.c)
            sol_domain.sol_int.srf = np.maximum(srf[:-1], srf[1:])

        # Dissipation term
        d_sol_prim = sol_prim_left - sol_prim_right
        sol_domain.roe_diss = self.calc_roe_diss(sol_ave)
        diss_term = 0.5 * (sol_domain.roe_diss * np.expand_dims(d_sol_prim, 0)).sum(-2)

        # Complete Roe flux
        flux = 0.5 * (flux_left + flux_right) + diss_term

        return flux

    def calc_roe_diss(self, sol_ave):
        """
        Compute dissipation term of Roe flux
        """

        gas_model = sol_ave.gas_model

        diss_matrix = np.zeros((gas_model.num_eqs, gas_model.num_eqs, sol_ave.num_cells), dtype=REAL_TYPE)

        # For clarity
        rho = sol_ave.sol_cons[0, :]
        press = sol_ave.sol_prim[0, :]
        vel = sol_ave.sol_prim[1, :]
        temp = sol_ave.sol_prim[2, :]
        mass_fracs = sol_ave.sol_prim[3:, :]

        # Derivatives of density and enthalpy
        (d_rho_d_press, d_rho_d_temp, d_rho_d_mass_frac) = gas_model.calc_dens_derivs(
            sol_ave.sol_cons[0, :],
            wrt_press=True,
            pressure=sol_ave.sol_prim[0, :],
            wrt_temp=True,
            temperature=sol_ave.sol_prim[2, :],
            wrt_spec=True,
            mix_mol_weight=sol_ave.mw_mix,
        )

        (d_enth_d_press, d_enth_d_temp, d_enth_d_mass_frac) = gas_model.calc_stag_enth_derivs(
            wrt_press=True,
            wrt_temp=True,
            mass_fracs=sol_ave.sol_prim[3:, :],
            wrt_spec=True,
            temperature=sol_ave.sol_prim[2, :],
        )

        # Save for Jacobian calculations
        sol_ave.d_rho_d_press = d_rho_d_press
        sol_ave.d_rho_d_temp = d_rho_d_temp
        sol_ave.d_rho_d_mass_frac = d_rho_d_mass_frac
        sol_ave.d_enth_d_press = d_enth_d_press
        sol_ave.d_enth_d_temp = d_enth_d_temp
        sol_ave.d_enth_d_mass_frac = d_enth_d_mass_frac

        # Gamma terms for energy equation
        g_press = rho * d_enth_d_press + d_rho_d_press * sol_ave.h0 - 1.0
        g_temp = rho * d_enth_d_temp + d_rho_d_temp * sol_ave.h0
        g_mass_frac = rho[None, :] * d_enth_d_mass_frac + sol_ave.h0[None, :] * d_rho_d_mass_frac

        # Characteristic speeds
        lambda1 = vel + sol_ave.c
        lambda2 = vel - sol_ave.c
        lambda1_abs = np.absolute(lambda1)
        lambda2_abs = np.absolute(lambda2)

        r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
        alpha = sol_ave.c * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
        beta = np.power(sol_ave.c, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
        phi = sol_ave.c * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

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
        for mf_idx_out in range(3, sol_ave.gas_model.num_eqs):
            for mf_idx_in in range(3, sol_ave.gas_model.num_eqs):
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

    def calc_jacob_prim(self, sol_domain):
        """
        Compute flux Jacobian with respect to the primitive variables
        """

        roe_diss = sol_domain.roe_diss.copy()

        jacob_left_face = self.calc_d_inv_flux_d_sol_prim(sol_domain.sol_left)
        jacob_right_face = self.calc_d_inv_flux_d_sol_prim(sol_domain.sol_right)

        # Jacobian wrt current cell
        jacob_center_cell = (jacob_left_face[:, :, 1:] + roe_diss[:, :, 1:]) + (
            -jacob_right_face[:, :, :-1] + roe_diss[:, :, :-1]
        )

        # Jacobian wrt left neighbor
        jacob_left_cell = -jacob_left_face[:, :, 1:-1] - roe_diss[:, :, :-2]

        # Jacobian wrt right neighbor
        jacob_right_cell = jacob_right_face[:, :, 1:-1] - roe_diss[:, :, 2:]

        return jacob_center_cell, jacob_left_cell, jacob_right_cell

import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.flux import Flux


class StandardViscFlux(Flux):
    """
    Standard viscous flux scheme with binary diffusion velocity approximation
    """

    def __init__(self, sol_domain):

        super().__init__()

    def calc_flux(self, sol_domain):
        """
        Compute flux array
        """

        gas = sol_domain.gas_model
        mesh = sol_domain.mesh
        sol_ave = sol_domain.sol_ave
        sol_prim_full = sol_domain.sol_prim_full

        # Compute 2nd-order state gradients at faces
        # TODO: generalize to higher orders of accuracy
        sol_prim_grad = np.zeros(
            (gas.num_eqs + 1, sol_domain.num_flux_faces), dtype=REAL_TYPE)
        sol_prim_grad[:-1, :] = (
            (sol_prim_full[:, sol_domain.flux_samp_right_idxs]
             - sol_prim_full[:, sol_domain.flux_samp_left_idxs])
            / mesh.dx)

        # Get gradient of last species for diffusion velocity term
        # TODO: maybe a sneakier way to do this?
        mass_fracs = gas.calc_all_mass_fracs(
            sol_prim_full[3:, :], threshold=False)
        sol_prim_grad[-1, :] = (
            (mass_fracs[-1, sol_domain.flux_samp_right_idxs]
             - mass_fracs[-1, sol_domain.flux_samp_left_idxs])
            / mesh.dx)

        # Thermo and transport props
        mole_fracs = gas.calc_all_mole_fracs(
            sol_ave.mass_fracs_full,
            mix_mol_weight=sol_ave.mw_mix)
        spec_dyn_visc = gas.calc_species_dynamic_visc(sol_ave.sol_prim[2, :])
        therm_cond_mix = gas.calc_mix_thermal_cond(
            spec_dyn_visc=spec_dyn_visc,
            mole_fracs=mole_fracs)
        dyn_visc_mix = gas.calc_mix_dynamic_visc(
            spec_dyn_visc=spec_dyn_visc,
            mole_fracs=mole_fracs)
        mass_diff_mix = gas.calc_species_mass_diff_coeff(
            sol_ave.sol_cons[0, :],
            spec_dyn_visc=spec_dyn_visc)
        hi = gas.calc_spec_enth(sol_ave.sol_prim[2, :])

        # Copy for use later
        sol_ave.dyn_visc_mix = dyn_visc_mix
        sol_ave.therm_cond_mix = therm_cond_mix
        sol_ave.mass_diff_mix = mass_diff_mix
        sol_ave.hi = hi

        # Stress "tensor"
        tau = 4.0 / 3.0 * dyn_visc_mix * sol_prim_grad[1, :]

        # Diffusion velocity
        diff_vel = (
            sol_ave.sol_cons[[0], :] * mass_diff_mix * sol_prim_grad[3:, :])

        # Correction velocity
        corr_vel = np.sum(diff_vel, axis=0)

        # Viscous flux
        flux_visc = np.zeros(
            (gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
        flux_visc[1, :] += tau
        flux_visc[2, :] += (sol_ave.sol_prim[1, :] * tau
                            + therm_cond_mix * sol_prim_grad[2, :]
                            + np.sum(diff_vel * hi, axis=0))
        flux_visc[3:, :] += (diff_vel[gas.mass_frac_slice]
                             - sol_ave.sol_prim[3:, :] * corr_vel[None, :])

        return flux_visc

    def calc_jacob_prim(self, sol_domain):
        """
        Compute flux Jacobian with respect to the primitive variables
        """

        # NOTE: signs are flipped to avoid an additional negation

        jacob_face = self.calc_d_visc_flux_d_sol_prim(sol_domain.sol_ave)

        # Jacobian wrt current cell
        jacob_center_cell = jacob_face[:, :, 1:] - jacob_face[:, :, :-1]

        # Jacobian wrt left neighbor
        jacob_left_cell = jacob_face[:, :, 1:-1]

        # Jacobian wrt right neighbor
        jacob_right_cell = -jacob_face[:, :, 1:-1]

        return jacob_center_cell, jacob_left_cell, jacob_right_cell

    def calc_d_visc_flux_d_sol_prim(self, sol_ave):
        """
        Compute Jacobian of viscous flux vector
        with respect to the primitive state

        sol_ave is the solutionPhys associated with
        the face state used to calculate the viscous flux
        """

        gas = sol_ave.gas_model

        d_flux_d_sol_prim = np.zeros(
            (gas.num_eqs, gas.num_eqs, sol_ave.num_cells))

        # momentum equation row
        d_flux_d_sol_prim[1, 1, :] = 4.0 / 3.0 * sol_ave.dyn_visc_mix

        # energy equation row
        d_flux_d_sol_prim[2, 1, :] = (
            4.0 / 3.0 * sol_ave.sol_prim[1, :] * sol_ave.dyn_visc_mix)
        d_flux_d_sol_prim[2, 2, :] = sol_ave.therm_cond_mix
        d_flux_d_sol_prim[2, 3:, :] = (
            sol_ave.sol_cons[[0], :]
            * (sol_ave.mass_diff_mix[gas.mass_frac_slice, :]
               * sol_ave.hi[gas.mass_frac_slice, :]
               - sol_ave.mass_diff_mix[[-1], :] * sol_ave.hi[[-1], :]))

        # species transport row
        # TODO: vectorize
        for i in range(3, gas.num_eqs):
            d_flux_d_sol_prim[i, i, :] = (
                sol_ave.sol_cons[0, :] * sol_ave.mass_diff_mix[i - 3, :])

        return d_flux_d_sol_prim

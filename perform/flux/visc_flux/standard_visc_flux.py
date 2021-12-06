import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.flux import Flux


class StandardViscFlux(Flux):
    """Standard viscous flux scheme with binary diffusion velocity approximation

    Inherits from Flux.

    Args:
        sol_domain: SolutionDomain with which this Flux is associated.
    """

    def __init__(self, sol_domain):

        super().__init__()

    def calc_flux(self, sol_domain):
        """Compute viscous flux vector.

        Computes state gradient at cell faces and computes viscous flux with respect to average face state.
        Assumes that average face state has already been computed.

        Args:
            sol_domain: SolutionDomain with which this Flux is associated.

        Returns:
            NumPy array of viscous flux for every governing equation at each finite volume face.
        """

        gas = sol_domain.gas_model
        mesh = sol_domain.mesh
        sol_ave = sol_domain.sol_ave
        sol_prim_full = sol_domain.sol_prim_full

        # Compute 2nd-order state gradients at faces
        # TODO: generalize to higher orders of accuracy
        sol_prim_grad = np.zeros((gas.num_eqs + 1, sol_domain.num_flux_faces), dtype=REAL_TYPE)
        sol_prim_grad[:-1, :] = (
            sol_prim_full[:, sol_domain.flux_samp_right_idxs] - sol_prim_full[:, sol_domain.flux_samp_left_idxs]
        ) / mesh.dx

        # Get gradient of last species for diffusion velocity term
        # TODO: pointless for single species, also sol_prim_grad is too big
        # TODO: maybe a sneakier way to do this?
        mass_fracs = gas.calc_all_mass_fracs(sol_prim_full[3:, :], threshold=False)
        sol_prim_grad[-1, :] = (
            mass_fracs[-1, sol_domain.flux_samp_right_idxs] - mass_fracs[-1, sol_domain.flux_samp_left_idxs]
        ) / mesh.dx

        # Stress "tensor"
        tau = 4.0 / 3.0 * sol_ave.dyn_visc_mix * sol_prim_grad[1, :]

        # Diffusion velocity
        diff_vel = sol_ave.sol_cons[[0], :] * sol_ave.mass_diff_mix * sol_prim_grad[3:, :]

        # Correction velocity
        corr_vel = np.sum(diff_vel, axis=0)

        # Viscous flux
        flux_visc = np.zeros((gas.num_eqs, sol_domain.num_flux_faces), dtype=REAL_TYPE)
        flux_visc[1, :] += tau
        flux_visc[2, :] += (
            sol_ave.sol_prim[1, :] * tau
            + sol_ave.therm_cond_mix * sol_prim_grad[2, :]
            + np.sum(diff_vel * sol_ave.hi, axis=0)
        )
        flux_visc[3:, :] += diff_vel[gas.mass_frac_slice] - sol_ave.sol_prim[3:, :] * corr_vel[None, :]

        return flux_visc

    def calc_jacob(self, sol_domain, wrt_prim):
        """Compute viscous flux Jacobian.

        Calculates analytical flux Jacobian at each face and assembles Jacobian with respect to each
        finite volume cell's state. Note that the gradient with respect to boundary ghost cell states are
        excluded, as the Newton iteration linear solve does not need this.

        Args:
            sol_domain: SolutionDomain with which this Flux is associated.
            wrt_prim:
                Boolean flag. If True, calculate Jacobian w/r/t the primitive variables.
                If False, calculate w/r/t conservative variables.

        Returns:
            jacob_center_cell: center block diagonal of flux Jacobian, representing the gradient of a given cell's
            viscous flux contribution with respect to its own primitive state.
            jacob_left_cell: lower block diagonal of flux Jacobian, representing the gradient of a given cell's
            viscous flux contribution with respect to its left neighbor's primitive state.
            jacob_left_cell: upper block diagonal of flux Jacobian, representing the gradient of a given cell's
            viscous flux contribution with respect to its right neighbor's primitive state.
        """

        center_samp = sol_domain.flux_rhs_idxs
        left_samp = sol_domain.jacob_left_samp
        right_samp = sol_domain.jacob_right_samp

        # NOTE: signs are flipped to avoid an additional negation

        # Jacobian of viscous flux vector at each face
        jacob_face = self.calc_d_visc_flux_d_sol_prim(sol_domain.sol_ave)
        # TODO: when conservative Jacobian is implemented, uncomment this
        # if wrt_prim:
        #     jacob_face = self.calc_d_visc_flux_d_sol_prim(sol_domain.sol_ave)
        # else:
        #     jacob_face = self.calc_d_visc_flux_d_sol_cons(sol_domain.sol_ave)

        # center, lower, and upper block diagonal of full numerical flux Jacobian
        jacob_center_cell = jacob_face[:, :, center_samp + 1] - jacob_face[:, :, center_samp]
        jacob_left_cell = jacob_face[:, :, left_samp]
        jacob_right_cell = -jacob_face[:, :, right_samp]

        return jacob_center_cell, jacob_left_cell, jacob_right_cell

    def calc_d_visc_flux_d_sol_prim(self, sol_ave):
        """Compute Jacobian of viscous flux vector with respect to the primitive state.

        This Jacobian is computed analytically at sol_ave, which is the "average" face state used
        to calculate the viscous flux.

        Please refer to the solver theory documentation for full derivations of this Jacobian.

        Args:
            sol_ave: SolutionPhys associated with the average state at each finite volume face.

        Returns:
            3D Jacobian of viscous flux Jacobian w/r/t the primitive state
        """

        gas = sol_ave.gas_model

        d_flux_d_sol_prim = np.zeros((gas.num_eqs, gas.num_eqs, sol_ave.num_cells))

        # momentum equation row
        d_flux_d_sol_prim[1, 1, :] = 4.0 / 3.0 * sol_ave.dyn_visc_mix

        # energy equation row
        d_flux_d_sol_prim[2, 1, :] = 4.0 / 3.0 * sol_ave.sol_prim[1, :] * sol_ave.dyn_visc_mix
        d_flux_d_sol_prim[2, 2, :] = sol_ave.therm_cond_mix
        d_flux_d_sol_prim[2, 3:, :] = sol_ave.sol_cons[[0], :] * (
            sol_ave.mass_diff_mix[gas.mass_frac_slice, :] * sol_ave.hi[gas.mass_frac_slice, :]
            - sol_ave.mass_diff_mix[[-1], :] * sol_ave.hi[[-1], :]
        )

        # species transport row
        # TODO: vectorize
        for i in range(3, gas.num_eqs):
            d_flux_d_sol_prim[i, i, :] = sol_ave.sol_cons[0, :] * sol_ave.mass_diff_mix[i - 3, :]

        return d_flux_d_sol_prim

    def calc_d_visc_flux_d_sol_cons(self, sol_ave):
        """Compute Jacobian of viscous flux vector with respect to the conservative state.

        Details coming when this is implemented!

        Args:
            sol_ave: SolutionPhys associated with the average state at each finite volume face.

        Returns:

        """

        raise ValueError("Conservative Jacobian not implemented yet.")

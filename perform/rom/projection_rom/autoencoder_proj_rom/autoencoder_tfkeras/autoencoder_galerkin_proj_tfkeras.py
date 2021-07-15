import numpy as np
from scipy.linalg import pinv

from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_tfkeras.autoencoder_tfkeras import AutoencoderTFKeras


class AutoencoderGalerkinProjTFKeras(AutoencoderTFKeras):
    """Class for projection-based ROM with a TF-Keras non-linear manifold decoder and Galerkin projection.

    Inherits from AutoencoderTFKeras.

    Decoder is assumed to map to the conservative variables. Allows implicit and explicit time integration.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if rom_domain.time_integrator.dual_time:
            raise ValueError("Galerkin is intended for conservative variable evolution, please set dual_time = False")

        super().__init__(model_idx, rom_domain, sol_domain)

    def calc_projector(self, sol_domain):
        """Compute RHS projection operator.

        Called by ProjectionROM.calc_rhs_low_dim() to compute projection operator which is applied to RHS function
        for explicit time integration.

        Decoder projector is pseudo-inverse of decoder Jacobian, and encoder projector is simply encoder Jacobian.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
        """

        jacob = self.calc_model_jacobian(sol_domain)

        if self.encoder_jacob:
            self.projector = jacob

        else:
            self.projector = pinv(jacob)

    def calc_d_code(self, res_jacob, res, sol_domain, rom_domain):
        """Compute change in low-dimensional state for implicit scheme Newton iteration.

        This function computes the iterative change in the low-dimensional state for a given Newton iteration
        of an implicit time integration scheme. Please refer to Lee and Carlberg (2020) for details on the
        formulation for Galerkin projection.

        Args:
            res_jacob:
                scipy.sparse.csr_matrix containing full-dimensional residual Jacobian with respect to
                the conservative variables.
            res: NumPy array of fully-discrete residual, already negated for Newton iteration.
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

        Returns:
            lhs: Left-hand side of low-dimensional linear solve.
            rhs: Right-hand side of low-dimensional linear solve.
        """

        # TODO: this is non-general and janky, only valid for BDF

        jacob = self.calc_model_jacobian(sol_domain)

        if self.encoder_jacob:
            jacob_pinv = jacob * self.norm_fac_prof_cons.ravel(order="C")[None, :]

        else:
            scaled_jacob = jacob * self.norm_fac_prof_cons.ravel(order="C")[:, None]
            jacob_pinv = pinv(scaled_jacob)

        # Newton iteration linear solve
        lhs = jacob_pinv @ ((res_jacob @ scaled_jacob) / self.norm_fac_prof_cons.ravel(order="C")[:, None])
        rhs = jacob_pinv @ (res / self.norm_fac_prof_cons).ravel(order="C")

        return lhs, rhs

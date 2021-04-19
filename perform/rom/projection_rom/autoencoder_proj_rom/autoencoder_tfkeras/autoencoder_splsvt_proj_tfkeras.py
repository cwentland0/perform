import numpy as np

from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_tfkeras.autoencoder_tfkeras import AutoencoderTFKeras


class AutoencoderSPLSVTProjTFKeras(AutoencoderTFKeras):
    """Class for projection-based ROM with a TF-Keras non-linear manifold decoder and SP-LSVT projection.

    Inherits from AutoencoderTFKeras. 
    
    Decoder is assumed to map to the primitive variables. Allows implicit time integration only.
    
    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        if rom_domain.time_integrator.time_type == "explicit":
            raise ValueError("Explicit NLM SP-LSVT not implemented yet")

        if (rom_domain.time_integrator.time_type == "implicit") and (not rom_domain.time_integrator.dual_time):
            raise ValueError(
                "NLM SP-LSVT is intended for primitive variable  evolution, please use Galerkin or LSPG,"
                + " or set dual_time = True"
            )

        super().__init__(model_idx, rom_domain, sol_domain)

        if self.encoder_jacob:
            raise ValueError(
                "SP-LSVT is not equipped with an encoder Jacobian approximation, please set encoder_jacob = False"
            )

    def calc_d_code(self, res_jacob, res, sol_domain):
        """Compute change in low-dimensional state for implicit scheme Newton iteration.

        This function computes the iterative change in the low-dimensional state for a given Newton iteration
        of an implicit time integration scheme.

        Args:
            res_jacob:
                scipy.sparse.csr_matrix containing full-dimensional residual Jacobian with respect to
                the primitive variables.
            res: NumPy array of fully-discrete residual, already negated for Newton iteration.
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

        Returns:
            d_code:
                Solution of low-dimensional linear solve, representing the iterative change in
                the low-dimensional state.
            lhs: Left-hand side of low-dimensional linear solve.
            rhs: Right-hand side of low-dimensional linear solve.
        """

        # decoder Jacobian, scaled
        jacob = self.calc_model_jacobian(sol_domain)
        scaled_jacob = jacob * self.norm_fac_prof_prim.ravel(order="C")[:, None]

        # test basis
        test_basis = (res_jacob @ scaled_jacob) / self.norm_fac_prof_cons.ravel(order="C")[:, None]

        # Newton iteration linear solve
        lhs = test_basis.T @ test_basis
        rhs = test_basis.T @ (res / self.norm_fac_prof_cons).ravel(order="C")

        d_code = np.linalg.solve(lhs, rhs)

        return d_code, lhs, rhs

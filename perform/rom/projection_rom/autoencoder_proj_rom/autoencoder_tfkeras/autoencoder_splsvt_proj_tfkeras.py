import numpy as np

from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_tfkeras.autoencoder_tfkeras import AutoencoderTFKeras


class AutoencoderSPLSVTProjTFKeras(AutoencoderTFKeras):
    """
    Class for computing non-linear Galerkin ROMs via a TensorFlow autoencoder
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
        """
        Compute change in low-dimensional state for
        implicit scheme Newton iteration
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

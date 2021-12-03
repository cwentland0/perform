import os

import numpy as np

from perform.rom.rom_space_mapping.rom_space_mapping import RomSpaceMapping


class LinearSpaceMapping(RomSpaceMapping):
    """Linear basis mapping to/from the state/latent spaces"""

    def __init__(self, sol_domain, rom_domain, rom_model):

        rom_dict = rom_domain.rom_dict

        # Input checking
        basis_files = rom_dict["basis_files"]
        assert len(basis_files) == rom_domain.num_models, "Must provide model_files for each model"  # redundant
        in_file = os.path.join(rom_domain.model_dir, basis_files[rom_model.model_idx])
        assert os.path.isfile(in_file), "Could not find basis file at " + in_file
        _, ext = os.path.splitext(in_file)
        assert ext == ".npy", "Basis file is not NumPy binary (*.npy): " + in_file
        self.basis_file = in_file

        super().__init__(sol_domain, rom_domain, rom_model)

        # Precompute scaled trial basis for implicit time integration
        if rom_domain.rom_method.is_intrusive:
            if rom_domain.time_stepper.time_integrator.time_type == "implicit":
                self.trial_basis_scaled = self.trial_basis * self.norm_fac_prof.ravel(order="C")[:, None]

    def load_mapping(self):

        # load and check trial basis
        self.trial_basis = np.load(self.basis_file)
        num_vars_basis, num_cells_basis, num_modes_basis = self.trial_basis.shape

        assert num_vars_basis == self.sol_shape[0], (
            "Basis at "
            + self.basis_file
            + " represents a different number of variables than specified by model_var_idxs ("
            + str(num_vars_basis)
            + " != "
            + str(self.sol_shape[0])
            + ")"
        )
        assert num_cells_basis == self.sol_shape[1], (
            "Basis at "
            + self.basis_file
            + " has a different number of cells ("
            + str(num_cells_basis)
            + " != "
            + str(self.sol_shape[1])
            + ")"
        )
        assert num_modes_basis >= self.latent_dim, (
            "Basis at "
            + self.basis_file
            + " must have at least "
            + str(self.latent_dim)
            + " modes ("
            + str(num_modes_basis)
            + " < "
            + str(self.latent_dim)
            + ")"
        )

        # flatten first two dimensions for easier matmul
        self.trial_basis = self.trial_basis[:, :, : self.latent_dim]
        self.trial_basis = np.reshape(self.trial_basis, (-1, self.latent_dim), order="C")

    def apply_encoder(self, sol):
        """Computes raw projection of solution vector"""

        sol = sol.ravel(order="C")
        code = self.trial_basis.T @ sol
        return code

    def apply_decoder(self, code):
        """Compute raw decoding of code.

        Only computes trial_basis @ code, does not compute any denormalization or decentering.

        Args:
            code: NumPy array of low-dimensional state, of dimension latent_dim.
        """

        sol = self.trial_basis @ code
        sol = np.reshape(sol, self.sol_shape, order="C")
        return sol

    def calc_decoder_jacob_pinv(self):
        """Returns pseudo-inverse of decoder Jacobian

        For linear, orthonormal trial basis, this is just the transpose of the trial basis
        # TODO: mapping may not always be orthonormal basis
        """

        return self.trial_basis.T

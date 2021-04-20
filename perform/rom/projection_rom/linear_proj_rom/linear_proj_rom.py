import numpy as np

from perform.rom.projection_rom.projection_rom import ProjectionROM


class LinearProjROM(ProjectionROM):
    """Base class for all linear subspace projection-based ROMs.

    Inherits from ProjectionROM. Assumes that solution decoding is computed by

    sol = cent_prof + norm_sub_prof + norm_fac_prof * (trial_basis @ code)

    Child classes must implement a calc_projector() member function if it permits explicit time integration,
    and/or a calc_d_code() member function if it permits implicit time integration.

    Args:
        model_idx: Zero-indexed ID of a given RomModel instance within a RomDomain's model_list.
        rom_domain: RomDomain within which this RomModel is contained.
        sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.

    Attributes:
        trial_basis:
            2D NumPy array containing latent_dim trial basis modes. Modes are flattened in C order,
            i.e. iterating first over cells, then over variables.
        hyper_reduc_dim: Number of modes to retain in hyper_reduc_basis.
        hyper_reduc_basis:
            2D NumPy array containing hyper_reduc_dim hyper-reduction basis modes. Modes are flattened in C order,
            i.e. iterating first over cells, then over variables.
        direct_samp_idxs_flat:
            NumPy array of slicing indices for slicing directly-sampled cells from
            solution-related vectors flattened in C order.
    """

    def __init__(self, model_idx, rom_domain, sol_domain):

        super().__init__(model_idx, rom_domain, sol_domain)

        # load and check trial basis
        self.trial_basis = np.load(rom_domain.model_files[self.model_idx])
        num_vars_basis_in, num_cells_basis_in, num_modes_basis_in = self.trial_basis.shape

        assert num_vars_basis_in == self.num_vars, (
            "Basis at "
            + rom_domain.model_files[self.model_idx]
            + " represents a different number of variables "
            + "than specified by modelVarIdxs ("
            + str(num_vars_basis_in)
            + " != "
            + str(self.num_vars)
            + ")"
        )
        assert num_cells_basis_in == sol_domain.mesh.num_cells, (
            "Basis at "
            + rom_domain.model_files[self.model_idx]
            + " has a different number of cells ("
            + str(num_cells_basis_in)
            + " != "
            + str(sol_domain.mesh.num_cells)
            + ")"
        )
        assert num_modes_basis_in >= self.latent_dim, (
            "Basis at "
            + rom_domain.model_files[self.model_idx]
            + " must have at least "
            + str(self.latent_dim)
            + " modes ("
            + str(num_modes_basis_in)
            + " < "
            + str(self.latent_dim)
            + ")"
        )

        # flatten first two dimensions for easier matmul
        self.trial_basis = self.trial_basis[:, :, : self.latent_dim]
        self.trial_basis = np.reshape(self.trial_basis, (-1, self.latent_dim), order="C")

        # load and check gappy POD basis
        if rom_domain.hyper_reduc:
            hyper_reduc_basis = np.load(rom_domain.hyper_reduc_files[self.model_idx])

            assert hyper_reduc_basis.ndim == 3, "Hyper-reduction basis must have three axes"
            assert hyper_reduc_basis.shape[:2] == (
                sol_domain.gas_model.num_eqs,
                sol_domain.mesh.num_cells,
            ), "Hyper reduction basis must have shape [num_eqs, num_cells, numHRModes]"

            self.hyper_reduc_dim = rom_domain.hyper_reduc_dims[self.model_idx]
            hyper_reduc_basis = hyper_reduc_basis[:, :, : self.hyper_reduc_dim]
            self.hyper_reduc_basis = np.reshape(hyper_reduc_basis, (-1, self.hyper_reduc_dim), order="C")

            # indices for sampling flattened hyper_reduc_basis
            self.direct_samp_idxs_flat = np.zeros(rom_domain.num_samp_cells * self.num_vars, dtype=np.int32)
            for var_num in range(self.num_vars):
                idx1 = var_num * rom_domain.num_samp_cells
                idx2 = (var_num + 1) * rom_domain.num_samp_cells
                self.direct_samp_idxs_flat[idx1:idx2] = (
                    rom_domain.direct_samp_idxs + var_num * sol_domain.mesh.num_cells
                )

        else:
            self.direct_samp_idxs_flat = np.s_[:]

    def init_from_sol(self, sol_domain):
        """Initialize full-order solution from projection of loaded full-order initial conditions.

        Computes L2 projection of the initial conditions onto the trial space, i.e. V * V^T * q_r, and sets as
        full-dimensional initial condition solution profile. This is automatically used if a low-dimensional state
        initial condition file is not provided.

        Args:
            sol_domain: SolutionDomain with which this RomModel's RomDomain is associated.
        """

        if self.target_cons:
            sol = self.scale_profile(
                sol_domain.sol_int.sol_cons[self.var_idxs, :],
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_cons,
                norm_sub_prof=self.norm_sub_prof_cons,
                center=True,
                cent_prof=self.cent_prof_cons,
                inverse=False,
            )

            self.code = self.project_to_low_dim(self.trial_basis, sol, transpose=True)
            sol_domain.sol_int.sol_cons[self.var_idxs, :] = self.decode_sol(self.code)

        else:
            sol = self.scale_profile(
                sol_domain.sol_int.sol_prim[self.var_idxs, :],
                normalize=True,
                norm_fac_prof=self.norm_fac_prof_prim,
                norm_sub_prof=self.norm_sub_prof_prim,
                center=True,
                cent_prof=self.cent_prof_prim,
                inverse=False,
            )

            self.code = self.project_to_low_dim(self.trial_basis, sol, transpose=True)
            sol_domain.sol_int.sol_prim[self.var_idxs, :] = self.decode_sol(self.code)

    def apply_decoder(self, code):
        """Compute raw decoding of code.

        Only computes trial_basis @ code, does not compute any denormalization or decentering.

        Args:
            code: NumPy array of low-dimensional state, of dimension latent_dim.
        """

        sol = self.trial_basis @ code
        sol = np.reshape(sol, (self.num_vars, -1), order="C")
        return sol

import numpy as np

from perform.constants import REAL_TYPE, R_UNIV

# TODO: some of the CPG functions can be generalized and placed here
# 	(e.g. calc sound speed in terms of enthalpy and density derivs)


class GasModel:
    """
    Base class storing constant chemical properties of modeled species
    Also includes universal gas methods (e.g. calc mixture molecular weight)
    """

    def __init__(self, gas_dict):

        # Gas composition
        self.num_species_full = int(gas_dict["num_species"])
        self.mol_weights = gas_dict["mol_weights"].astype(REAL_TYPE)

        # Species names for plotting and output
        try:
            self.species_names = gas_dict["species_names"]
            assert (len(self.species_names) == self.num_species_full)
        except KeyError:
            self.species_names = [
                "Species_" + str(x + 1) for x in range(self.num_species_full)]

        # Check input lengths
        assert (len(self.mol_weights) == self.num_species_full)
        # TODO: check lengths of reaction inputs

        # Dealing with single-species option
        if self.num_species_full == 1:
            self.num_species = self.num_species_full
        else:
            self.num_species = self.num_species_full - 1

        self.mass_frac_slice = np.arange(self.num_species)
        self.mw_inv = 1.0 / self.mol_weights
        self.mw_inv_diffs = self.mw_inv[self.mass_frac_slice] - self.mw_inv[-1]

        self.num_eqs = self.num_species + 3

        # Mass matrices for calculating
        # 	viscosity and thermal conductivity mixing laws
        self.mix_mass_matrix = np.zeros(
            (self.num_species_full, self.num_species_full), dtype=REAL_TYPE)
        self.mix_inv_mass_matrix = np.zeros(
            (self.num_species_full, self.num_species_full), dtype=REAL_TYPE)
        self.precomp_mix_mass_matrices()

    def precomp_mix_mass_matrices(self):
        """
        Precompute mass matrices for dynamic viscosity mixing law
        """

        for spec_idx in range(self.num_species_full):
            self.mix_mass_matrix[spec_idx, :] = np.power(
                (self.mol_weights / self.mol_weights[spec_idx]),
                0.25)
            self.mix_inv_mass_matrix[spec_idx, :] = (
                (1.0 / (2.0 * np.sqrt(2.0)))
                * (1.0 / np.sqrt(1.0 + self.mol_weights[spec_idx]
                                 / self.mol_weights)))

    def get_mass_frac_array(self, sol_prim=None, mass_fracs=None):
        """
        Helper function to handle array slicing
        to avoid weird NumPy array broadcasting issues
        """

        # Get all but last mass fraction field
        if (sol_prim is None):
            assert (mass_fracs is not None), (
                "Must provide mass fractions"
                + " if not providing primitive solution")
            if (mass_fracs.ndim == 1):
                mass_fracs = np.reshape(mass_fracs, (1, -1))

            if (mass_fracs.shape[0] == self.num_species_full):
                mass_fracs = mass_fracs[self.mass_frac_slice, :]
            else:
                assert (mass_fracs.shape[0] == self.num_species), (
                    "If not passing full mass fraction array,"
                    + " must pass N-1 species")
        else:
            mass_fracs = sol_prim[3:, :]

        return mass_fracs

    def calc_all_mass_fracs(self, mass_fracs_ns, threshold=True):
        """
        Helper function to compute all num_species_full
        mass fraction fields from num_species fields

        Thresholds all mass fraction fields between zero and unity
        """

        if (self.num_species_full == 1):
            mass_fracs = np.maximum(0.0, np.minimum(1.0, mass_fracs_ns))
        else:
            num_species, num_cells = mass_fracs_ns.shape
            assert (num_species == self.num_species), (
                "mass_fracs_ns argument must have "
                + str(self.num_species) + " species")

            mass_fracs = np.zeros(
                (num_species + 1, num_cells), dtype=REAL_TYPE)
            if threshold:
                mass_fracs[:-1, :] = np.maximum(
                    0.0, np.minimum(1.0, mass_fracs_ns))
                mass_fracs[-1, :] = 1.0 - np.sum(mass_fracs[:-1, :], axis=0)
                mass_fracs[-1, :] = np.maximum(
                    0.0, np.minimum(1.0, mass_fracs[-1, :]))
            else:
                mass_fracs[:-1, :] = mass_fracs_ns
                mass_fracs[-1, :] = 1.0 - np.sum(mass_fracs[:-1, :], axis=0)

        return mass_fracs

    def calc_mix_mol_weight(self, mass_fracs):
        """
        Compute mixture molecular weight
        """

        if (mass_fracs.shape[0] == self.num_species):
            mass_fracs = self.calc_all_mass_fracs(mass_fracs, threshold=False)

        mix_mol_weight = (
            1.0
            / np.sum(mass_fracs / self.mol_weights[:, None], axis=0))

        return mix_mol_weight

    def calc_all_mole_fracs(self, mass_fracs, mix_mol_weight=None):
        """
        Compute mole fractions of all species from mass fractions
        """

        if (mass_fracs.shape[0] == self.num_species):
            mass_fracs = self.calc_all_mass_fracs(mass_fracs, threshold=False)

        if (mix_mol_weight is None):
            mix_mol_weight = self.calc_mix_mol_weight(mass_fracs)

        mole_fracs = (
            mass_fracs * mix_mol_weight[None, :] * self.mw_inv[:, None])

        return mole_fracs

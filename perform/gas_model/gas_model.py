import numpy as np

from perform.constants import REAL_TYPE, R_UNIV

# TODO: some of the CPG functions can be generalized and placed here
# 	(e.g. calc sound speed in terms of enthalpy and density derivs)


class GasModel:
    """Base class for all gas models (e.g. CPG, TPG).
    
    GasModel reads various universal gas properties (e.g. species molecular weights) from the chemistry input file
    and implements several universal gas methods (e.g. calculating mixture molecular weight).

    Child classes must implement the following member methods:

    * calc_mix_gas_constant()
    * calc_mix_gamma()
    * calc_mix_cp()
    * calc_spec_enth()
    * calc_stag_enth()
    * calc_species_dynamic_visc()
    * calc_mix_dynamic_visc()
    * calc_species_therm_cond()
    * calc_mix_thermal_cond()
    * calc_species_mass_diff_coeff()
    * calc_sound_speed()
    * calc_dens_derivs()
    * calc_stag_enth_derivs()
    
    See calorically_perfect_gas.py for details on each.

    Args:
        chem_dict: Dictionary of input parameters read from chemistry file.

    Attributes:
        num_species_full: Total number of chemical species to be modeled.
        num_species: If num_species_full == 1, num_species == 1. Otherwise, num_species == num_species_full - 1
        species_names:
            List of strings, names of chemical species. If not provided by user, defaults to "Species X",
            where X is the species index.
        mol_weights: NumPy array of chemical species molecular weights in g/mol
        mass_frac_slice: Range for slicing mass fraction arrays, avoids issues when num_species_full == 1
        mw_inv: NumPy array of reciprocals of molecular weights, used in some calculations
        mw_inv_diffs:
            NumPy array of difference in molecular weights between the first num_species species and the last species,
            used in some calculations.
        num_eqs:
            Number of governing equations in system.
            Equal to 3 + num_species (continuity, momentum, energy, and species transport).
        mix_mass_matrix: NumPy array used in computing mixture dynamic viscosity.
        mix_inv_mass_matrix: NumPy array used in computing mixture dynamic viscosity.
    """

    def __init__(self, chem_dict):

        # Gas composition
        self.num_species_full = int(chem_dict["num_species"])
        self.mol_weights = chem_dict["mol_weights"].astype(REAL_TYPE)

        # Species names for plotting and output
        try:
            self.species_names = chem_dict["species_names"]
            assert len(self.species_names) == self.num_species_full
        except KeyError:
            self.species_names = ["Species_" + str(x + 1) for x in range(self.num_species_full)]

        # Check input lengths
        assert len(self.mol_weights) == self.num_species_full

        # Dealing with single-species option
        if self.num_species_full == 1:
            self.num_species = self.num_species_full
        else:
            self.num_species = self.num_species_full - 1

        self.mass_frac_slice = np.arange(self.num_species)
        self.mw_inv = 1.0 / self.mol_weights
        self.mw_inv_diffs = self.mw_inv[self.mass_frac_slice] - self.mw_inv[-1]

        self.num_eqs = self.num_species + 3

        # Mass matrices for calculating viscosity and thermal conductivity mixing laws
        self.mix_mass_matrix = np.zeros((self.num_species_full, self.num_species_full), dtype=REAL_TYPE)
        self.mix_inv_mass_matrix = np.zeros((self.num_species_full, self.num_species_full), dtype=REAL_TYPE)
        for spec_idx in range(self.num_species_full):
            self.mix_mass_matrix[spec_idx, :] = np.power((self.mol_weights / self.mol_weights[spec_idx]), 0.25)
            self.mix_inv_mass_matrix[spec_idx, :] = (1.0 / (2.0 * np.sqrt(2.0))) * (
                1.0 / np.sqrt(1.0 + self.mol_weights[spec_idx] / self.mol_weights)
            )

    def get_mass_frac_array(self, sol_prim_in=None, mass_fracs_in=None):
        """Helper function to get num_species mass fraction vectors.
        
        This function helps to avoid weird NumPy array broadcasting issues, namely the automatic
        squeezing of singleton dimensions. Can retrieve num_species mass fraction fields from
        sol_prim_in or mass_fracs_in.

        Args:
            sol_prim_in:
                NumPy array of primitive solution profile (e.g. from SolutionPhys),
                including pressure, velocity, temperature, and species mass fractions.
            mass_fracs_in:
                NumPy array of species mass fractions. May pass array of num_species or num_species_full profiles.

        Returns:
            NumPy array containing num_species mass fraction profiles.
        """

        # Get all but last mass fraction field
        if sol_prim_in is None:
            assert mass_fracs_in is not None, "Must provide mass fractions if not providing primitive solution"
            if mass_fracs_in.ndim == 1:
                mass_fracs = np.reshape(mass_fracs_in, (1, -1)).copy()
            else:
                mass_fracs = mass_fracs_in.copy()

            if mass_fracs.shape[0] == self.num_species_full:
                mass_fracs = mass_fracs[self.mass_frac_slice, :]
            else:
                assert (
                    mass_fracs.shape[0] == self.num_species
                ), "If not passing full mass fraction array, must pass N-1 species"
        else:
            assert sol_prim_in is not None, "Must provide primitive solution if not providing mass fractions"
            mass_fracs = sol_prim_in[3:, :].copy()

        return mass_fracs

    def calc_all_mass_fracs(self, mass_fracs_ns, threshold=True):
        """Helper function to compute all num_species_full mass fraction fields.

        This function takes advantage of the fact that all species mass fractions at a given location must be equal
        to unity. Thus, the mass fraction of the last chemical species (which is not directly modeled) is equal to
        unity minus the sum of the other mass fractions. 
        
        Additionally, if threshold=True, all mass fraction fields are thresholded between zero and unity.

        In the case num_species_full == 1, simply thesholds between zero and unity.

        Args:
            mass_fracs_ns: NumPy array of num_species mass fraction fields.
            threshold: Boolean flag indicating whether to threshold all species mass fractions between zero and unity.

        Returns:
            NumPy array of num_species_full mass fraction fields, all summing to unity in each row.
        """

        if self.num_species_full == 1:
            mass_fracs = np.maximum(0.0, np.minimum(1.0, mass_fracs_ns))
        else:
            num_species, num_cells = mass_fracs_ns.shape
            assert num_species == self.num_species, (
                "mass_fracs_ns argument must have " + str(self.num_species) + " species"
            )

            mass_fracs = np.zeros((num_species + 1, num_cells), dtype=REAL_TYPE)
            if threshold:
                mass_fracs[:-1, :] = np.maximum(0.0, np.minimum(1.0, mass_fracs_ns))
                mass_fracs[-1, :] = 1.0 - np.sum(mass_fracs[:-1, :], axis=0)
                mass_fracs[-1, :] = np.maximum(0.0, np.minimum(1.0, mass_fracs[-1, :]))
            else:
                mass_fracs[:-1, :] = mass_fracs_ns
                mass_fracs[-1, :] = 1.0 - np.sum(mass_fracs[:-1, :], axis=0)

        return mass_fracs

    def calc_mix_mol_weight(self, mass_fracs):
        """Compute mixture molecular weights.

        Args:
            mass_fracs:
                NumPy array of species mass fraction fields. If num_species fields provided, calculates the final
                num_species_full-th mass fraction field.

        Returns:
            NumPy array of mixture molecular weights.
        """

        if mass_fracs.shape[0] == self.num_species:
            mass_fracs = self.calc_all_mass_fracs(mass_fracs, threshold=False)

        mix_mol_weight = 1.0 / np.sum(mass_fracs / self.mol_weights[:, None], axis=0)

        return mix_mol_weight

    def calc_all_mole_fracs(self, mass_fracs, mix_mol_weight=None):
        """Compute species mole fractions of all species.

        Args:
            mass_fracs:
                NumPy array of species mass fraction fields. If num_species fields provided, calculates the final
                num_species_full-th mass fraction field.
            mix_mol_weight: Mixture molecular weight. If not provided, is calculated.

        Returns:
            NumPy array of mole fraction fields for all num_species_full chemical species.
        """

        if mass_fracs.shape[0] == self.num_species:
            mass_fracs = self.calc_all_mass_fracs(mass_fracs, threshold=False)

        if mix_mol_weight is None:
            mix_mol_weight = self.calc_mix_mol_weight(mass_fracs)

        mole_fracs = mass_fracs * mix_mol_weight[None, :] * self.mw_inv[:, None]

        return mole_fracs

import unittest

import numpy as np

from perform.constants import REAL_TYPE
from perform.gas_model.gas_model import GasModel


class GasModelInitTestCase(unittest.TestCase):
    def setUp(self):

        self.num_species = 3
        self.mol_weights = np.array([32.0, 28.0, 40.0], dtype=REAL_TYPE)
        self.species_names = np.array(["oxygen", "nitrogen", "argon"])

        self.chem_dict = {}
        self.chem_dict["num_species"] = self.num_species
        self.chem_dict["mol_weights"] = self.mol_weights
        self.chem_dict["species_names"] = self.species_names

    def test_gas_model_init(self):

        gas = GasModel(self.chem_dict)

        # check inputs
        self.assertEqual(gas.num_species_full, self.num_species)
        self.assertTrue(np.array_equal(gas.mol_weights, self.mol_weights))
        self.assertTrue(np.array_equal(gas.species_names, self.species_names))

        # check other attributes
        self.assertEqual(gas.num_species, self.num_species - 1)
        self.assertTrue(np.array_equal(gas.mass_frac_slice, np.array([0, 1])))
        self.assertAlmostEqual(gas.mw_inv[0], 0.03125)
        self.assertAlmostEqual(gas.mw_inv[1], 0.0357142857)
        self.assertAlmostEqual(gas.mw_inv[2], 0.025)
        self.assertAlmostEqual(gas.mw_inv_diffs[0], 0.00625)
        self.assertAlmostEqual(gas.mw_inv_diffs[1], 0.0107142857)
        self.assertEqual(gas.num_eqs, 5)


class GasModelMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.num_species = 3
        self.mol_weights = np.array([32.0, 28.0, 40.0], dtype=REAL_TYPE)
        self.species_names = np.array(["oxygen", "nitrogen", "argon"])

        self.chem_dict = {}
        self.chem_dict["num_species"] = self.num_species
        self.chem_dict["mol_weights"] = self.mol_weights
        self.chem_dict["species_names"] = self.species_names

        self.mass_fracs = np.array([[0.75511, 0.75511], [0.2314, 0.2314], [0.01349, 0.01349]], dtype=REAL_TYPE)

        self.gas = GasModel(self.chem_dict)

    def test_get_mass_frac_array(self):

        # check with all mass fractions
        mass_fracs_out = self.gas.get_mass_frac_array(mass_fracs_in=self.mass_fracs)
        self.assertTrue(np.array_equal(mass_fracs_out, np.array([[0.75511, 0.75511], [0.2314, 0.2314]])))

        # check with N - 1 mass fractions
        mass_fracs_out = self.gas.get_mass_frac_array(mass_fracs_in=self.mass_fracs[:-1, :])
        self.assertTrue(np.array_equal(mass_fracs_out, np.array([[0.75511, 0.75511], [0.2314, 0.2314]])))

    def test_calc_all_mass_fracs(self):

        # check without thresholding
        mass_fracs_out = self.gas.calc_all_mass_fracs(self.mass_fracs[:-1, :], threshold=False)
        self.assertTrue(np.allclose(mass_fracs_out, self.mass_fracs))

        # check with thresholding, should do nothing
        mass_fracs_out = self.gas.calc_all_mass_fracs(self.mass_fracs[:-1, :], threshold=True)
        self.assertTrue(np.allclose(mass_fracs_out, self.mass_fracs))

    def test_calc_mix_mol_weight(self):

        mix_mol_weight = self.gas.calc_mix_mol_weight(self.mass_fracs)
        self.assertTrue(np.allclose(mix_mol_weight, np.array([31.0571321, 31.0571321])))

    def test_calc_all_mole_fracs(self):

        mole_fracs = self.gas.calc_all_mole_fracs(self.mass_fracs)
        self.assertTrue(np.allclose(mole_fracs, np.array([[0.73286097, 0.73286097], [0.25666501, 0.25666501], [0.010474018, 0.010474018]])))

if __name__ == "__main__":
    unittest.main()

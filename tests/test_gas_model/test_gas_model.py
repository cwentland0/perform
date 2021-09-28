import unittest

import numpy as np

from perform.constants import REAL_TYPE
from perform.gas_model.gas_model import GasModel
from constants import CHEM_DICT_AIR

class GasModelInitTestCase(unittest.TestCase):
    def setUp(self):

        self.chem_dict = CHEM_DICT_AIR

    def test_gas_model_init(self):

        gas = GasModel(self.chem_dict)

        # check inputs
        self.assertEqual(gas.num_species_full, self.chem_dict["num_species"])
        self.assertTrue(np.array_equal(gas.mol_weights, self.chem_dict["mol_weights"]))
        self.assertTrue(np.array_equal(gas.species_names, self.chem_dict["species_names"]))

        # check other attributes
        self.assertEqual(gas.num_species, self.chem_dict["num_species"] - 1)
        self.assertTrue(np.array_equal(gas.mass_frac_slice, np.array([0, 1])))
        self.assertAlmostEqual(gas.mw_inv[0], 0.03125)
        self.assertAlmostEqual(gas.mw_inv[1], 0.0357142857)
        self.assertAlmostEqual(gas.mw_inv[2], 0.025)
        self.assertAlmostEqual(gas.mw_inv_diffs[0], 0.00625)
        self.assertAlmostEqual(gas.mw_inv_diffs[1], 0.0107142857)
        self.assertEqual(gas.num_eqs, 5)

        mix_mass_matrix = np.array(
            [[1.0, 0.9671682, 1.057371], [1.0339463, 1.0, 1.0932651], [0.9457416, 0.9146912, 1.0]]
        )
        self.assertTrue(np.allclose(gas.mix_mass_matrix, mix_mass_matrix))

        mix_inv_mass_matrix = np.array(
            [[0.25, 0.2415229, 0.2635231], [0.2581989, 0.25, 0.27116307], [0.2357023, 0.2268713, 0.25]]
        )
        self.assertTrue(np.allclose(gas.mix_inv_mass_matrix, mix_inv_mass_matrix))


class GasModelMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.chem_dict = CHEM_DICT_AIR

        self.mass_fracs = np.repeat(np.array([[0.231537], [0.755063], [0.0134]], dtype=REAL_TYPE), repeats=2, axis=1)

        self.gas = GasModel(self.chem_dict)

    def test_get_mass_frac_array(self):

        # check with all mass fractions
        mass_fracs_out = self.gas.get_mass_frac_array(mass_fracs_in=self.mass_fracs)
        self.assertTrue(
            np.array_equal(
                mass_fracs_out, np.repeat(np.array([[0.231537], [0.755063]], dtype=REAL_TYPE), repeats=2, axis=1)
            )
        )

        # check with N - 1 mass fractions
        mass_fracs_out = self.gas.get_mass_frac_array(mass_fracs_in=self.mass_fracs[:-1, :])
        self.assertTrue(
            np.array_equal(
                mass_fracs_out, np.repeat(np.array([[0.231537], [0.755063]], dtype=REAL_TYPE), repeats=2, axis=1)
            )
        )

    def test_calc_all_mass_fracs(self):

        # check without thresholding
        mass_fracs_out = self.gas.calc_all_mass_fracs(self.mass_fracs[:-1, :], threshold=False)
        self.assertTrue(np.allclose(mass_fracs_out, self.mass_fracs))

        # check with thresholding, should do nothing
        mass_fracs_out = self.gas.calc_all_mass_fracs(self.mass_fracs[:-1, :], threshold=True)
        self.assertTrue(np.allclose(mass_fracs_out, self.mass_fracs))

    def test_calc_mix_mol_weight(self):

        mix_mol_weight = self.gas.calc_mix_mol_weight(self.mass_fracs)
        self.assertTrue(
            np.allclose(mix_mol_weight, np.repeat(np.array([[28.9543985]], dtype=REAL_TYPE), repeats=2, axis=1))
        )

    def test_calc_all_mole_fracs(self):

        mole_fracs = self.gas.calc_all_mole_fracs(self.mass_fracs)
        self.assertTrue(
            np.allclose(
                mole_fracs,
                np.repeat(np.array([[0.20950046], [0.78079982], [0.00969972]], dtype=REAL_TYPE), repeats=2, axis=1),
            )
        )


if __name__ == "__main__":
    unittest.main()

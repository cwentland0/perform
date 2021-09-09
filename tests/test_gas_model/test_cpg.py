import unittest

import numpy as np

from perform.constants import REAL_TYPE
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas

class CPGInitTestCase(unittest.TestCase):
    def setUp(self):

        self.num_species = 3
        self.mol_weights = np.array([32.0, 28.0, 40.0], dtype=REAL_TYPE)
        self.species_names = np.array(["oxygen", "nitrogen", "argon"])
        self.enth_ref = np.array([0.0, 0.0, 0.0], dtype=REAL_TYPE)
        self.cp = np.array([918.0, 1040.0, 520.3], dtype=REAL_TYPE)
        self.pr = np.array([0.730, 0.718, 0.687], dtype=REAL_TYPE)
        self.sc = np.array([0.612, 0.612, 0.612], dtype=REAL_TYPE)
        self.mu_ref = np.array([2.07e-5, 1.76e-5, 2.27e-5], dtype=REAL_TYPE)
        self.temp_ref = np.array([0.0, 0.0, 0.0], dtype=REAL_TYPE)

        self.chem_dict = {}
        self.chem_dict["num_species"] = self.num_species
        self.chem_dict["mol_weights"] = self.mol_weights
        self.chem_dict["species_names"] = self.species_names
        self.chem_dict["enth_ref"] = self.enth_ref
        self.chem_dict["cp"] = self.cp
        self.chem_dict["pr"] = self.pr
        self.chem_dict["sc"] = self.sc
        self.chem_dict["mu_ref"] = self.mu_ref
        self.chem_dict["temp_ref"] = self.temp_ref

    def test_cpg_init(self):

        gas = CaloricallyPerfectGas(self.chem_dict)

        # check inputs
        self.assertTrue(np.array_equal(gas.cp, self.cp))
        self.assertTrue(np.array_equal(gas.pr, self.pr))
        self.assertTrue(np.array_equal(gas.sc, self.sc))
        self.assertTrue(np.array_equal(gas.mu_ref, self.mu_ref))
        self.assertTrue(np.array_equal(gas.temp_ref, self.temp_ref))

        # check other quantities
        self.assertTrue(np.array_equal(gas.const_visc_idxs, np.array([0, 1, 2])))
        self.assertTrue(np.array_equal(gas.suth_visc_idxs, np.array([])))
        self.assertTrue(np.allclose(gas.cp_diffs, np.array([397.7, 519.7])))
        self.assertTrue(np.allclose(gas.enth_ref_diffs, np.array([0.0, 0.0])))


class GasModelMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.num_species = 3
        self.mol_weights = np.array([32.0, 28.0, 40.0], dtype=REAL_TYPE)
        self.species_names = np.array(["oxygen", "nitrogen", "argon"])
        self.enth_ref = np.array([0.0, 0.0, 0.0], dtype=REAL_TYPE)
        self.cp = np.array([918.0, 1040.0, 520.3], dtype=REAL_TYPE)
        self.pr = np.array([0.730, 0.718, 0.687], dtype=REAL_TYPE)
        self.sc = np.array([0.612, 0.612, 0.612], dtype=REAL_TYPE)
        self.mu_ref = np.array([2.07e-5, 1.76e-5, 2.27e-5], dtype=REAL_TYPE)
        self.temp_ref = np.array([0.0, 0.0, 0.0], dtype=REAL_TYPE)

        self.chem_dict = {}
        self.chem_dict["num_species"] = self.num_species
        self.chem_dict["mol_weights"] = self.mol_weights
        self.chem_dict["species_names"] = self.species_names
        self.chem_dict["enth_ref"] = self.enth_ref
        self.chem_dict["cp"] = self.cp
        self.chem_dict["pr"] = self.pr
        self.chem_dict["sc"] = self.sc
        self.chem_dict["mu_ref"] = self.mu_ref
        self.chem_dict["temp_ref"] = self.temp_ref

        self.sol_prim = np.repeat(np.array([[1e6], [10.0], [300]], dtype=REAL_TYPE), repeats=2, axis=1)
        self.mass_fracs = np.repeat(np.array([[0.231537], [0.755063], [0.0134]], dtype=REAL_TYPE), repeats=2, axis=1)
        self.sol_prim = np.concatenate((self.sol_prim, self.mass_fracs), axis=0)

        self.gas = CaloricallyPerfectGas(self.chem_dict)

    def test_calc_mix_gas_constant(self):

        r_mix = self.gas.calc_mix_gas_constant(self.mass_fracs)
        self.assertTrue(np.allclose(r_mix, np.array([[287.1571343], [287.1571343]])))

    def test_calc_mix_enth_ref(self):

        enth_ref_mix = self.gas.calc_mix_enth_ref(self.mass_fracs)
        self.assertTrue(np.allclose(enth_ref_mix, np.array([[0.0], [0.0]])))

    def test_calc_mix_cp(self):

        cp_mix = self.gas.calc_mix_cp(self.mass_fracs)
        self.assertTrue(np.allclose(cp_mix, np.array([[1004.788506], [1004.788506]])))

    def test_calc_mix_gamma(self):

        gamma_mix = self.gas.calc_mix_gamma(mass_fracs=self.mass_fracs)
        self.assertTrue(np.allclose(gamma_mix, np.array([[1.4001457], [1.4001457]])))

    def test_calc_density(self):

        density = self.gas.calc_density(self.sol_prim)
        self.assertTrue(np.allclose(density, np.array([[11.6080464], [11.6080464]])))

    def test_calc_spec_enth(self):

        spec_enth = self.gas.calc_spec_enth(self.sol_prim[2, :])
        self.assertTrue(np.allclose(spec_enth, np.repeat(np.array([[275400], [312000], [156090]]), repeats=2, axis=1)))

    def test_calc_stag_enth(self):

        stag_enth = self.gas.calc_stag_enth(self.sol_prim[1, :], self.mass_fracs, temperature=self.sol_prim[2, :])
        self.assertTrue(np.allclose(stag_enth, np.array([[301486.55], [301486.55]])))

    def test_calc_species_dynamic_visc(self):

        spec_dyn_visc = self.gas.calc_species_dynamic_visc(self.sol_prim[2, :])
        self.assertTrue(np.allclose(spec_dyn_visc, np.repeat(np.array([[2.07e-5], [1.76e-5], [2.27e-5]]), repeats=2, axis=1)))

    def test_calc_mix_dynamic_visc(self):

        mix_dyn_visc = self.gas.calc_mix_dynamic_visc(temperature=self.sol_prim[2, :], mass_fracs=self.mass_fracs)
        self.assertTrue(np.allclose(mix_dyn_visc, np.array([1.8300075e-5, 1.8300075e-5,])))

    def test_calc_species_therm_cond(self):

        spec_therm_cond = self.gas.calc_species_therm_cond(temperature=self.sol_prim[2, :])
        self.assertTrue(np.allclose(spec_therm_cond, np.repeat(np.array([[0.02603096], [0.02549303], [0.01719186]]), repeats=2, axis=1)))

    def test_calc_mix_thermal_cond(self):

        mix_therm_cond = self.gas.calc_mix_thermal_cond(temperature=self.sol_prim[2, :], mass_fracs=self.mass_fracs)
        self.assertTrue(np.allclose(mix_therm_cond, np.array([0.02550461, 0.02550461])))

    def test_calc_species_mass_diff_coeff(self):

        density = self.gas.calc_density(self.sol_prim)
        spec_mass_diff = self.gas.calc_species_mass_diff_coeff(density, temperature=self.sol_prim[2, :])
        self.assertTrue(np.allclose(spec_mass_diff, np.repeat(np.array([[2.9138003e-6], [2.4774341e-6], [3.195327e-6]]), repeats=2, axis=1)))

    def test_calc_sound_speed(self):

        sound_speed = self.gas.calc_sound_speed(self.sol_prim[2, :], mass_fracs=self.mass_fracs)
        self.assertTrue(np.allclose(sound_speed, np.array([347.30181, 347.30181])))

    def test_calc_dens_derivs(self):

        density = self.gas.calc_density(self.sol_prim)
        dens_derivs = self.gas.calc_dens_derivs(
            density,
            wrt_press=True,
            pressure=self.sol_prim[0, :],
            wrt_temp=True,
            temperature=self.sol_prim[2, :],
            wrt_spec=True,
            mass_fracs=self.mass_fracs,
        )

        self.assertTrue(np.allclose(dens_derivs[0], np.array([1.160804e-5, 1.160804e-5])))
        self.assertTrue(np.allclose(dens_derivs[1], np.array([-0.03869349, -0.03869349])))
        self.assertTrue(np.allclose(dens_derivs[2], np.repeat(np.array([[-2.10065], [-3.601114]]), repeats=2, axis=1)))

    def test_calc_stag_enth_derivs(self):

        stag_enth_derivs = self.gas.calc_stag_enth_derivs(
            wrt_press=True,
            wrt_temp=True,
            mass_fracs=self.mass_fracs,
            wrt_vel=True,
            velocity=self.sol_prim[1, :],
            wrt_spec=True,
            temperature=self.sol_prim[2, :],
        )

        self.assertTrue(np.allclose(stag_enth_derivs[0], np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(stag_enth_derivs[1], np.array([[1004.788506], [1004.788506]])))
        self.assertTrue(np.allclose(stag_enth_derivs[2], self.sol_prim[1, :]))
        self.assertTrue(np.allclose(stag_enth_derivs[3], np.repeat(np.array([[119310.0], [155910.0]]), repeats=2, axis=1)))

    # def test_calc_press_temp_from_cons(self):
    # TODO: this is an iterative procedure, not sure how to test this easily

if __name__ == "__main__":
    unittest.main()

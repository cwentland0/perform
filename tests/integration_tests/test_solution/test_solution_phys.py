import unittest
import os

import numpy as np

from constants import CHEM_DICT_AIR
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_phys import SolutionPhys


class SolutionPhysInitTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        self.chem_dict = CHEM_DICT_AIR
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        self.sol_prim_in = np.array(
            [
                [1e6, 1e5],
                [2.0, 1.0],
                [300.0, 400.0],
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        )

        self.sol_cons_in = np.array([[11.8, 0.9], [23.6, 0.9], [2.45e6, 2.45e5], [4.72, 0.54], [7.08, 0.36]])

        self.num_cells = 2

    def test_solution_phys_init(self):

        # NOTE: SolutionPhys initialization incidentally tests update_state

        # check initialization from primitive state
        sol = SolutionPhys(self.gas, self.num_cells, sol_prim_in=self.sol_prim_in)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_phys_init_sol_cons.npy"), sol.sol_cons)

        else:

            self.assertTrue(np.array_equal(sol.sol_prim, self.sol_prim_in))
            self.assertTrue(
                np.allclose(sol.sol_cons, np.load(os.path.join(self.output_dir, "sol_phys_init_sol_cons.npy")))
            )

            # TODO: a LOT of checking of other variables

        # check initialization from conservative state (just check that primitive state is correct this time)
        sol = SolutionPhys(self.gas, self.num_cells, sol_cons_in=self.sol_cons_in)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_phys_init_sol_prim.npy"), sol.sol_prim)

        else:

            self.assertTrue(np.array_equal(sol.sol_cons, self.sol_cons_in))
            self.assertTrue(
                np.allclose(sol.sol_prim, np.load(os.path.join(self.output_dir, "sol_phys_init_sol_prim.npy")))
            )


class SolutionPhysMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        self.chem_dict = CHEM_DICT_AIR
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        self.sol_prim_in = np.array(
            [
                [1e6, 1e5],
                [2.0, 1.0],
                [300.0, 400.0],
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        )

        self.num_cells = 2

        self.sol = SolutionPhys(self.gas, self.num_cells, self.sol_prim_in)

    def test_update_density_enthalpy_derivs(self):

        self.sol.update_density_enthalpy_derivs()

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "d_rho_d_press.npy"), self.sol.d_rho_d_press)
            np.save(os.path.join(self.output_dir, "d_rho_d_temp.npy"), self.sol.d_rho_d_temp)
            np.save(os.path.join(self.output_dir, "d_rho_d_mass_frac.npy"), self.sol.d_rho_d_mass_frac)
            np.save(os.path.join(self.output_dir, "d_enth_d_press.npy"), self.sol.d_enth_d_press)
            np.save(os.path.join(self.output_dir, "d_enth_d_temp.npy"), self.sol.d_enth_d_temp)
            np.save(os.path.join(self.output_dir, "d_enth_d_mass_frac.npy"), self.sol.d_enth_d_mass_frac)

        else:

            # check density derivatives
            self.assertTrue(
                np.allclose(self.sol.d_rho_d_press, np.load(os.path.join(self.output_dir, "d_rho_d_press.npy")))
            )
            self.assertTrue(
                np.allclose(self.sol.d_rho_d_temp, np.load(os.path.join(self.output_dir, "d_rho_d_temp.npy")))
            )
            self.assertTrue(
                np.allclose(self.sol.d_rho_d_mass_frac, np.load(os.path.join(self.output_dir, "d_rho_d_mass_frac.npy")))
            )

            # check enthalpy derivatives
            self.assertTrue(
                np.allclose(self.sol.d_enth_d_press, np.load(os.path.join(self.output_dir, "d_enth_d_press.npy")))
            )
            self.assertTrue(
                np.allclose(self.sol.d_enth_d_temp, np.load(os.path.join(self.output_dir, "d_enth_d_temp.npy")))
            )
            self.assertTrue(
                np.allclose(
                    self.sol.d_enth_d_mass_frac, np.load(os.path.join(self.output_dir, "d_enth_d_mass_frac.npy"))
                )
            )

    def test_calc_state_from_rho_h0(self):

        # slightly modify stagnation enthalpy and density
        self.sol.h0 = np.array([290000.0, 380000.0])
        self.sol.sol_cons[0, :] = np.array([11.8, 0.9])

        self.sol.calc_state_from_rho_h0()

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "sol_prim_from_rho_h0.npy"), self.sol.sol_prim)
            np.save(os.path.join(self.output_dir, "sol_cons_from_rho_h0.npy"), self.sol.sol_cons)

        else:

            # check new state
            self.assertTrue(
                np.allclose(self.sol.sol_prim, np.load(os.path.join(self.output_dir, "sol_prim_from_rho_h0.npy")))
            )

            self.assertTrue(
                np.allclose(self.sol.sol_cons, np.load(os.path.join(self.output_dir, "sol_cons_from_rho_h0.npy")))
            )

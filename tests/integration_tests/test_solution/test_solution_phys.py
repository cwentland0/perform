import unittest

import numpy as np

from constants import CHEM_DICT_AIR
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_phys import SolutionPhys


class SolutionPhysInitTestCase(unittest.TestCase):
    def setUp(self):

        self.chem_dict = CHEM_DICT_AIR
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [300.0, 400.0],
            [0.4, 0.6],
            [0.6, 0.4],
        ])

        self.sol_cons_in = np.array([
            [11.8, 0.9],
            [23.6, 0.9],
            [2.45e6, 2.45e5],
            [4.72, 0.54],
            [7.08, 0.36]
        ])

        self.num_cells = 2

    def test_solution_phys_init(self):

        # NOTE: SolutionPhys initialization incidentally tests update_state

        # check initialization from primitive state
        sol = SolutionPhys(self.gas, self.num_cells, sol_prim_in=self.sol_prim_in)
        self.assertTrue(np.allclose(
            sol.sol_cons,
            np.array([
                [11.8162321, 0.910169230],
                [23.6324642, 0.910169230],
                [2.51369841e6, 2.51981100e5],
                [4.72649284, 0.546101538],
                [7.08973927, 0.364067692],
            ])
        ))

        # TODO: a LOT of checking of other variables

        # check initialization from conservative state (just check that primitive state is correct this time)
        sol = SolutionPhys(self.gas, self.num_cells, sol_cons_in=self.sol_cons_in)
        self.assertTrue(np.allclose(
            sol.sol_prim,
            np.array([
                [9.74659260e5, 9.72295116e4],
                [2, 1],
                [292.800001, 393.312487],
                [0.4, 0.6],
                [0.6, 0.4],
            ])
        ))

class SolutionPhysMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.chem_dict = CHEM_DICT_AIR
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        self.sol_prim_in = np.array([
            [1e6, 1e5],
            [2.0, 1.0],
            [300.0, 400.0],
            [0.4, 0.6],
            [0.6, 0.4],
        ])

        self.num_cells = 2

        self.sol = SolutionPhys(self.gas, self.num_cells, self.sol_prim_in)

    def test_update_density_enthalpy_derivs(self):

        self.sol.update_density_enthalpy_derivs()

        # check density derivatives
        self.assertTrue(np.allclose(
            self.sol.d_rho_d_press,
            np.array([1.18162321e-5, 9.10169230e-6]),
        ))
        self.assertTrue(np.allclose(
            self.sol.d_rho_d_temp,
            np.array([-0.03938744, -0.00227542]),
        ))
        self.assertTrue(np.allclose(
            self.sol.d_rho_d_mass_frac,
            np.array([[-2.17667434, -0.17219418], [-3.73144172, -0.29519002]]),
        ))

        # check enthalpy derivatives
        self.assertTrue(np.allclose(
            self.sol.d_enth_d_press,
            np.array([0., 0.]),
        ))
        self.assertTrue(np.allclose(
            self.sol.d_enth_d_temp,
            np.array([991.2, 966.8]),
        ))
        self.assertTrue(np.allclose(
            self.sol.d_enth_d_mass_frac,
            np.array([[119310., 159080.], [155910., 207880.]]),
        ))

    def test_calc_state_from_rho_h0(self):

        # slightly modify stagnation enthalpy and density
        self.sol.h0 = np.array([290000. , 380000.])
        self.sol.sol_cons[0, :] = np.array([11.8,  0.9])

        self.sol.calc_state_from_rho_h0()
        
        # check new state
        self.assertTrue(np.allclose(
            self.sol.sol_prim,
            np.array([
                [9.73902428e5, 9.71643058e4],
                [2, 1],
                [2.92572639e2, 3.93048717e2],
                [0.4, 0.6],
                [0.6, 0.4]
            ])
        ))

        self.assertTrue(np.allclose(
            self.sol.sol_cons,
            np.array([
                [11.8, 0.9],
                [23.6, 0.9],
                [2.44809757e6, 2.44835694e5],
                [4.72, 0.54],
                [7.08, 0.36]
            ])
        ))

import unittest
import os

import numpy as np

from constants import CHEM_DICT_AIR
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_phys import SolutionPhys
from perform.flux.invisc_flux.invisc_flux import InviscFlux


class InviscFluxMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

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
        self.flux_model = InviscFlux()

    def test_calc_inv_flux(self):
        
        flux = self.flux_model.calc_inv_flux(self.sol.sol_cons, self.sol.sol_prim, self.sol.h0)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "invisc_flux.npy"), flux)

        else:

            self.assertTrue(np.allclose(
                flux,
                np.load(os.path.join(self.output_dir, "invisc_flux.npy"))
            ))

    def test_calc_flux_jacob(self):
        
        jacob = self.flux_model.calc_d_inv_flux_d_sol_prim(self.sol)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "d_inv_flux_d_sol_prim.npy"), jacob)

        else:

            self.assertTrue(np.allclose(
                jacob,
                np.load(os.path.join(self.output_dir, "d_inv_flux_d_sol_prim.npy"))
            ))
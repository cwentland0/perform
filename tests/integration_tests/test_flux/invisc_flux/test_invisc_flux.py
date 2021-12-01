import unittest
import os

import numpy as np

from constants import get_output_mode, CHEM_DICT_AIR, SOL_PRIM_IN_AIR
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.solution.solution_phys import SolutionPhys
from perform.flux.invisc_flux.invisc_flux import InviscFlux


class InviscFluxMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        self.chem_dict = CHEM_DICT_AIR
        self.gas = CaloricallyPerfectGas(self.chem_dict)

        self.num_cells = 2
        self.sol = SolutionPhys(self.gas, self.num_cells, SOL_PRIM_IN_AIR)
        self.flux_model = InviscFlux()

    def test_calc_inv_flux(self):

        flux = self.flux_model.calc_inv_flux(self.sol.sol_cons, self.sol.sol_prim, self.sol.h0)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "invisc_flux.npy"), flux)

        else:

            self.assertTrue(np.allclose(flux, np.load(os.path.join(self.output_dir, "invisc_flux.npy"))))

    def test_calc_flux_jacob(self):

        jacob = self.flux_model.calc_d_inv_flux_d_sol_prim(self.sol)

        if self.output_mode:

            np.save(os.path.join(self.output_dir, "d_inv_flux_d_sol_prim.npy"), jacob)

        else:

            self.assertTrue(np.allclose(jacob, np.load(os.path.join(self.output_dir, "d_inv_flux_d_sol_prim.npy"))))

import unittest

import numpy as np

from constants import TEST_DIR, del_test_dir, gen_test_dir, solution_domain_setup, rom_domain_setup, SOL_PRIM_IN_REACT
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.rom.rom_domain import RomDomain


class RomDomainInitTestCase(unittest.TestCase):
    def setUp(self):

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="mplsvt", space_mapping="linear", var_mapping="primitive")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

    def tearDown(self):

        del_test_dir()

    def test_rom_domain_init(self):

        rom_domain = RomDomain(self.sol_domain, self.solver)

        self.assertEqual(rom_domain.num_models, 1)
        self.assertTrue(np.array_equal(rom_domain.latent_dims, [8]))
        self.assertEqual(rom_domain.latent_dim_total, 8)
        self.assertTrue(np.array_equal(self.sol_domain.sol_int.sol_prim, SOL_PRIM_IN_REACT))

        # TODO: could probably check some other stuff

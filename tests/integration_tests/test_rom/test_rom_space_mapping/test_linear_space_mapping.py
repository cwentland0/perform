import unittest
import os

import numpy as np

from constants import (
    SOL_PRIM_IN_REACT,
    TEST_DIR,
    del_test_dir,
    gen_test_dir,
    solution_domain_setup,
    rom_domain_setup,
    get_output_mode,
)
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.rom.rom_domain import RomDomain


class LinearSpaceMappingInitTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

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

    def test_linear_space_mapping_init(self):

        rom_domain = RomDomain(self.sol_domain, self.solver)
        mapping = rom_domain.model_list[0].space_mapping

        basis_file = os.path.join(TEST_DIR, "model_files", "spatial_modes.npy")
        self.assertEqual(mapping.basis_file, basis_file)
        self.assertTrue(np.allclose(mapping.trial_basis, np.eye(8)))
        self.assertTrue(np.allclose(mapping.trial_basis_scaled, np.eye(8)))


class LinearSpaceMappingMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="mplsvt", space_mapping="linear", var_mapping="primitive")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)
        self.rom_domain = RomDomain(self.sol_domain, self.solver)
        self.mapping = self.rom_domain.model_list[0].space_mapping

    def tearDown(self):

        del_test_dir()

    def test_apply_mapping(self):

        code = self.mapping.apply_encoder(SOL_PRIM_IN_REACT)
        self.assertTrue(np.allclose(code, SOL_PRIM_IN_REACT.ravel(order="C")))

        sol = self.mapping.apply_decoder(code)
        self.assertTrue(np.allclose(sol, SOL_PRIM_IN_REACT))

    def test_calc_decoder_jacob(self):

        jacob = self.mapping.calc_decoder_jacob(None)
        self.assertTrue(
            np.allclose(
                jacob,
                self.mapping.trial_basis,
            )
        )

        jacob_pinv = self.mapping.calc_decoder_jacob_pinv(None)
        self.assertTrue(
            np.allclose(
                jacob_pinv,
                self.mapping.trial_basis.T,
            )
        )

        jacob_pinv = self.mapping.calc_decoder_jacob_pinv(None, jacob=jacob)
        self.assertTrue(
            np.allclose(
                jacob_pinv,
                jacob.T,
            )
        )

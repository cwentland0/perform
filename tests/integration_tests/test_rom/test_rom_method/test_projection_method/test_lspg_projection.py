import unittest

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


class LSPGProjectionLinearMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="lspg", space_mapping="linear", var_mapping="conservative")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.solver.param_dict["dual_time"] = False
        self.sol_domain = SolutionDomain(self.solver)
        self.rom_domain = RomDomain(self.sol_domain, self.solver)
        self.rom_method = self.rom_domain.rom_method

    def tearDown(self):

        del_test_dir()

    def test_init(self):

        self.assertTrue(self.rom_method.is_intrusive)
        self.assertTrue(np.allclose(self.rom_method.trial_basis_concat, np.eye(8)))
        self.assertTrue(np.allclose(self.rom_method.trial_basis_scaled_concat, np.eye(8)))

    def test_calc_jacobs(self):

        decoder_jacob_concat, scaled_decoder_jacob_concat = self.rom_method.assemble_concat_decoder_jacobs(
            self.sol_domain, self.rom_domain
        )
        self.assertTrue(np.allclose(decoder_jacob_concat, np.eye(8)))
        self.assertTrue(np.allclose(scaled_decoder_jacob_concat, np.eye(8)))

        decoder_jacob_pinv = self.rom_method.calc_concat_jacob_pinv(self.rom_domain, decoder_jacob_concat)
        self.assertTrue(np.allclose(decoder_jacob_pinv, np.eye(8)))

    def test_calc_d_code(self):

        res = SOL_PRIM_IN_REACT.copy()
        res_jacob = np.repeat(SOL_PRIM_IN_REACT.ravel(order="C")[:, None], 8, axis=-1)

        lhs, rhs = self.rom_method.calc_d_code(res_jacob, res, self.sol_domain, self.rom_domain)

        self.assertTrue(np.allclose(lhs, res_jacob.T @ res_jacob))
        self.assertTrue(np.allclose(rhs, res_jacob.T @ res.ravel(order="C")))


class LSPGProjectionAutoencoderMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="lspg", space_mapping="autoencoder", var_mapping="conservative")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.solver.param_dict["dual_time"] = False
        self.sol_domain = SolutionDomain(self.solver)
        self.rom_domain = RomDomain(self.sol_domain, self.solver)
        self.rom_method = self.rom_domain.rom_method

    def tearDown(self):

        del_test_dir()

    def test_init(self):

        self.assertTrue(self.rom_method.is_intrusive)

    def test_calc_jacobs(self):

        decoder_jacob_concat, scaled_decoder_jacob_concat = self.rom_method.assemble_concat_decoder_jacobs(
            self.sol_domain, self.rom_domain
        )
        self.assertTrue(np.allclose(decoder_jacob_concat, np.eye(8)))
        self.assertTrue(np.allclose(scaled_decoder_jacob_concat, np.eye(8)))

        decoder_jacob_pinv = self.rom_method.calc_concat_jacob_pinv(self.rom_domain, decoder_jacob_concat)
        self.assertTrue(np.allclose(decoder_jacob_pinv, np.eye(8)))

    def test_calc_d_code(self):

        res = SOL_PRIM_IN_REACT.copy()
        res_jacob = np.repeat(SOL_PRIM_IN_REACT.ravel(order="C")[:, None], 8, axis=-1)

        lhs, rhs = self.rom_method.calc_d_code(res_jacob, res, self.sol_domain, self.rom_domain)

        self.assertTrue(np.allclose(lhs, res_jacob.T @ res_jacob))
        self.assertTrue(np.allclose(rhs, res_jacob.T @ res.ravel(order="C")))

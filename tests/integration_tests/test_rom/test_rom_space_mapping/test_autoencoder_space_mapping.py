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
import tensorflow as tf
from keras.engine.functional import Functional
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.rom.rom_domain import RomDomain
from perform.rom.ml_library.tfkeras_library import TFKerasLibrary


class AutoencoderSpaceMappingInitTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="mplsvt", space_mapping="autoencoder", var_mapping="primitive")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

    def tearDown(self):

        del_test_dir()

    def test_autoencoder_space_mapping_init(self):

        rom_domain = RomDomain(self.sol_domain, self.solver)
        mapping = rom_domain.model_list[0].space_mapping

        encoder_file = os.path.join(TEST_DIR, "model_files", "encoder.h5")
        decoder_file = os.path.join(TEST_DIR, "model_files", "decoder.h5")
        self.assertEqual(mapping.encoder_file, encoder_file)
        self.assertEqual(mapping.decoder_file, decoder_file)

        # check objects
        self.assertTrue(isinstance(mapping.mllib, TFKerasLibrary))
        self.assertTrue(isinstance(mapping.encoder, Functional))
        self.assertTrue(isinstance(mapping.decoder, Functional))

        # check I/O shapes/dtypes
        self.assertEqual(mapping.encoder_io_shapes[0], (8,))
        self.assertEqual(mapping.encoder_io_shapes[1], (8,))
        self.assertEqual(mapping.encoder_io_dtypes[0], "float32")
        self.assertEqual(mapping.encoder_io_dtypes[1], "float32")
        self.assertEqual(mapping.decoder_io_shapes[0], (8,))
        self.assertEqual(mapping.decoder_io_shapes[1], (8,))
        self.assertEqual(mapping.decoder_io_dtypes[0], "float32")
        self.assertEqual(mapping.decoder_io_dtypes[1], "float32")

        # test decoder Jacobian input persistent memort
        self.assertTrue(isinstance(mapping.jacob_input, tf.Variable))
        self.assertEqual(mapping.jacob_input.shape, (1, 8))
        self.assertEqual(mapping.jacob_input.dtype, "float32")


class AutoencoderSpaceMappingMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="mplsvt", space_mapping="autoencoder", var_mapping="primitive")

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

        jacob = self.mapping.calc_decoder_jacob(SOL_PRIM_IN_REACT.ravel(order="C"))
        self.assertTrue(
            np.allclose(
                jacob,
                np.eye(8),
            )
        )

        jacob_pinv = self.mapping.calc_decoder_jacob_pinv(SOL_PRIM_IN_REACT.ravel(order="C"))
        self.assertTrue(
            np.allclose(
                jacob_pinv,
                np.eye(8),
            )
        )

        jacob_pinv = self.mapping.calc_decoder_jacob_pinv(SOL_PRIM_IN_REACT.ravel(order="C"), jacob=jacob)
        self.assertTrue(
            np.allclose(
                jacob_pinv,
                np.eye(8),
            )
        )

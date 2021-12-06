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


class TFKerasLibraryMethodsTestCase(unittest.TestCase):
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
        self.mllib = self.rom_domain.mllib
        self.encoder = self.rom_domain.model_list[0].space_mapping.encoder

    def tearDown(self):

        del_test_dir()

    def test_init_device(self):

        self.mllib.init_device(False)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")

        # TODO: add run_gpu=True test when you figure out how to run tests on GPU

    def test_load_model_obj(self):

        model_file = os.path.join(TEST_DIR, "model_files", "encoder.h5")
        model = self.mllib.load_model_obj(model_file)
        self.assertTrue(isinstance(model, Functional))

    def test_init_persistent_mem(self):

        mem_obj = self.mllib.init_persistent_mem((5, 6), dtype="float32", prepend_batch=False)
        self.assertTrue(isinstance(mem_obj, tf.Variable))
        self.assertEqual(mem_obj.shape, (5, 6))
        self.assertEqual(mem_obj.dtype, "float32")

        # check prepending dummy batch size
        mem_obj = self.mllib.init_persistent_mem((5, 6), dtype="float64", prepend_batch=True)
        self.assertTrue(isinstance(mem_obj, tf.Variable))
        self.assertEqual(mem_obj.shape, (1, 5, 6))
        self.assertEqual(mem_obj.dtype, "float64")

    def test_check_model_io(self):

        encoder_io_shapes, encoder_io_dtypes = self.mllib.check_model_io(
            self.encoder, (8,), (8,), isconv=False, io_format=None
        )
        self.assertEqual(encoder_io_shapes[0], (8,))
        self.assertEqual(encoder_io_shapes[1], (8,))
        self.assertEqual(encoder_io_dtypes[0], "float32")
        self.assertEqual(encoder_io_dtypes[1], "float32")

    def test_infer_model(self):

        inference = self.mllib.infer_model(self.encoder, SOL_PRIM_IN_REACT.ravel(order="C"))
        self.assertTrue(np.allclose(inference, SOL_PRIM_IN_REACT.ravel(order="C")[None, :]))

    def test_calc_model_jacobian(self):

        persistent_mem = self.mllib.init_persistent_mem((8,), "float32", prepend_batch=True)

        jacob = self.mllib.calc_model_jacobian(
            self.encoder, SOL_PRIM_IN_REACT.ravel(order="C"), output_shape=(8,), persistent_input=persistent_mem
        )

        self.assertTrue(np.allclose(jacob, np.eye(8)))

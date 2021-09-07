import unittest

import numpy as np

import perform.constants as constants


class ConstantsTestCase(unittest.TestCase):
    """Very simple tests for contants that are used throughout the code."""

    def test_types(self):

        self.assertIn(constants.REAL_TYPE, [np.float32, np.float64])
        self.assertIn(constants.COMPLEX_TYPE, [np.complex64, np.complex128])

    def test_floats(self):

        self.assertTrue(constants.R_UNIV > 0.0)
        self.assertTrue(constants.SUTH_TEMP > 0.0)
        self.assertTrue(constants.TINY_NUM > 0.0)
        self.assertTrue(constants.HUGE_NUM > 0.0)
        self.assertTrue(constants.L2_RES_TOL_DEFAULT > 0.0)
        self.assertTrue(constants.L2_STEADY_TOL_DEFAULT > 0.0)
        self.assertTrue(constants.DTAU_DEFAULT > 0.0)
        self.assertTrue(constants.CFL_DEFAULT > 0.0)
        self.assertTrue(constants.VNN_DEFAULT > 0.0)
        self.assertTrue(constants.FD_STEP_DEFAULT > 0.0)
        self.assertTrue(constants.FIG_WIDTH_DEFAULT > 0.0)
        self.assertTrue(constants.FIG_HEIGHT_DEFAULT > 0.0)

    def test_ints(self):

        self.assertTrue(constants.SUBITER_MAX_IMP_DEFAULT >= 1)

    def test_lists(self):

        # primitive residual scaling
        self.assertTrue(constants.RES_NORM_PRIM_DEFAULT[0] > 0.0)
        self.assertTrue(constants.RES_NORM_PRIM_DEFAULT[1] != 0.0)
        self.assertTrue(constants.RES_NORM_PRIM_DEFAULT[2] > 0.0)
        for idx in range(3, len(constants.RES_NORM_PRIM_DEFAULT)):
            self.assertTrue(constants.RES_NORM_PRIM_DEFAULT[idx] > 0.0)
            self.assertTrue(constants.RES_NORM_PRIM_DEFAULT[idx] <= 1.0)

    def test_strings(self):

        # check that directories and input files are strings
        self.assertIsInstance(constants.UNSTEADY_OUTPUT_DIR_NAME, str)
        self.assertIsInstance(constants.PROBE_OUTPUT_DIR_NAME, str)
        self.assertIsInstance(constants.IMAGE_OUTPUT_DIR_NAME, str)
        self.assertIsInstance(constants.RESTART_OUTPUT_DIR_NAME, str)
        self.assertIsInstance(constants.PARAM_INPUTS, str)
        self.assertIsInstance(constants.ROM_INPUTS, str)

        # check that they're at least valid as files/directories
        self.assertTrue(len(constants.UNSTEADY_OUTPUT_DIR_NAME) > 0)
        self.assertTrue(len(constants.PROBE_OUTPUT_DIR_NAME) > 0)
        self.assertTrue(len(constants.IMAGE_OUTPUT_DIR_NAME) > 0)
        self.assertTrue(len(constants.RESTART_OUTPUT_DIR_NAME) > 0)
        self.assertTrue(len(constants.PARAM_INPUTS) > 0)
        self.assertTrue(len(constants.ROM_INPUTS) > 0)


if __name__ == "__main__":
    unittest.main()

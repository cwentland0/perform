import unittest

import numpy as np

from perform.time_integrator.explicit_integrator import ExplicitIntegrator, RKExplicit, ClassicRK4, SSPRK3, JamesonLowStore


class ExplicitTimeIntInitTestCase(unittest.TestCase):
    def setUp(self):

        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "classic_rk4"
        self.param_dict["time_order"] = 3

    def test_explicit_time_int_init(self):

        time_int = ExplicitIntegrator(self.param_dict)
        
        self.assertEqual(time_int.time_type, "explicit")
        self.assertFalse(time_int.dual_time)
        self.assertFalse(time_int.adapt_dtau)


class ClassicRK4InitTestCase(unittest.TestCase):
    def setUp(self):
        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "classic_rk4"
        self.param_dict["time_order"] = 4

    def test_classic_rk4_init(self):

        time_int = ClassicRK4(self.param_dict)

        self.assertEqual(time_int.subiter_max, 4)
        self.assertEqual(len(time_int.rk_rhs), 4)

        # TODO: Is this really how I should be testing? The format of these could reasonably change.
        self.assertEqual(time_int.rk_a_vals.shape, (4, 4))
        self.assertEqual(time_int.rk_b_vals.shape, (4,))
        self.assertEqual(time_int.rk_c_vals.shape, (4,))


class ClassicRK4MethodsTestCase(unittest.TestCase):
    def setUp(self):
        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "classic_rk4"
        self.param_dict["time_order"] = 4

        # set up time integrator
        self.time_int = ClassicRK4(self.param_dict)
        self.time_int.subiter = 0

        self.rhs = np.array([[0, 3], [1, 4], [2, 5]])

    def test_solve_sol_change_subiter(self):

        self.time_int.rk_rhs[0] = self.rhs.copy()
        dsol = self.time_int.solve_sol_change_subiter(self.rhs)
        self.assertTrue(np.array_equal(dsol, np.array([[0, 1.5], [0.5, 2], [1, 2.5]])))

    # def test_solve_sol_change_iter(self):

    # def test_solve_sol_change(self):


class SSPRK3InitTestCase(unittest.TestCase):
    def setUp(self):
        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "ssp_rk3"
        self.param_dict["time_order"] = 3

    def test_ssp_rk3_init(self):

        time_int = SSPRK3(self.param_dict)

        self.assertEqual(time_int.subiter_max, 3)
        self.assertEqual(len(time_int.rk_rhs), 3)
        self.assertEqual(time_int.rk_a_vals.shape, (3, 3))
        self.assertEqual(time_int.rk_b_vals.shape, (3,))
        self.assertEqual(time_int.rk_c_vals.shape, (3,))

# TODO: Jameson low storage
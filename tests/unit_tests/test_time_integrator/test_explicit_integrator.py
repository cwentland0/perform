import unittest

import numpy as np

from perform.time_integrator.explicit_integrator import (
    ExplicitIntegrator,
    RKExplicit,
    ClassicRK4,
    SSPRK3,
    JamesonLowStore,
)


class ExplicitTimeIntInitTestCase(unittest.TestCase):
    def setUp(self):

        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "classic_rk4"
        self.param_dict["time_order"] = 4

    def test_explicit_time_int_init(self):

        time_int = ExplicitIntegrator(self.param_dict)

        self.assertEqual(time_int.time_type, "explicit")
        self.assertFalse(time_int.dual_time)
        self.assertFalse(time_int.adapt_dtau)

class RKExplicitInitTestCase(unittest.TestCase):
    def setUp(self):

        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "classic_rk4"
        self.param_dict["time_order"] = 4

    def test_rk_explicit_int_init(self):

        time_int = RKExplicit(self.param_dict)

        self.assertEqual(len(time_int.rk_rhs), 4)

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

        self.rhs = np.array([[0, 3], [1, 4], [2, 5]])
        self.rk_rhs = [
            np.array([[0, 3], [1, 4], [2, 5]]),
            np.array([[1, 4], [2, 5], [3, 6]]),
            np.array([[2, 5], [3, 6], [3, 7]]),
            np.array([[3, 6], [4, 7], [4, 8]]),
        ]

    def test_solve_sol_change_subiter(self):

        self.time_int.rk_rhs[0] = self.rhs.copy()
        dsol = self.time_int.solve_sol_change_subiter(self.rhs)
        self.assertTrue(np.allclose(dsol, np.array([[0, 1.5], [0.5, 2], [1, 2.5]])))

    def test_solve_sol_change_iter(self):

        self.time_int.rk_rhs = self.rk_rhs
        dsol = self.time_int.solve_sol_change_iter(self.rhs)
        self.assertTrue(np.allclose(dsol, np.array([[1.5, 4.5], [2.5, 5.5], [3, 6.5]])))

    def test_solve_sol_change(self):

        for self.time_int.subiter in range(self.time_int.subiter_max):

            dsol = self.time_int.solve_sol_change(self.rk_rhs[self.time_int.subiter])

            if self.time_int.subiter == 0:
                self.assertTrue(np.allclose(dsol, np.array([[0, 1.5e-7], [5e-8, 2e-7], [1e-7, 2.5e-7]])))
            elif self.time_int.subiter == 1:
                self.assertTrue(np.allclose(dsol, np.array([[5.0e-8, 2.0e-7], [1.0e-7, 2.5e-7], [1.5e-07, 3.0e-7]])))
            elif self.time_int.subiter == 2:
                self.assertTrue(np.allclose(dsol, np.array([[2.0e-7, 5.e-7], [3.0e-7, 6.0e-7], [3.0e-7, 7.0e-7]])))
            elif self.time_int.subiter == 3:
                self.assertTrue(np.allclose(dsol, np.array([[1.5e-7, 4.5e-7], [2.5e-7, 5.5e-7], [3e-7, 6.5e-7]])))


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


class SSPRK3MethodsTestCase(unittest.TestCase):
    def setUp(self):
        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "ssp_rk3"
        self.param_dict["time_order"] = 3

        # set up time integrator
        self.time_int = SSPRK3(self.param_dict)

        self.rhs = np.array([[0, 3], [1, 4], [2, 5]])
        self.rk_rhs = [
            np.array([[0, 3], [1, 4], [2, 5]]),
            np.array([[1, 4], [2, 5], [3, 6]]),
            np.array([[2, 5], [3, 6], [3, 7]]),
        ]

    def test_solve_sol_change_subiter(self):

        self.time_int.rk_rhs[0] = self.rhs.copy()
        dsol = self.time_int.solve_sol_change_subiter(self.rhs)
        self.assertTrue(np.allclose(dsol, np.array([[0, 3], [1, 4], [2, 5]])))

    def test_solve_sol_change_iter(self):

        self.time_int.rk_rhs = self.rk_rhs
        dsol = self.time_int.solve_sol_change_iter(self.rhs)
        self.assertTrue(np.allclose(dsol, np.array([[1.5, 4.5], [2.5, 5.5], [17/6, 6.5]])))

    def test_solve_sol_change(self):

        for self.time_int.subiter in range(self.time_int.subiter_max):

            dsol = self.time_int.solve_sol_change(self.rk_rhs[self.time_int.subiter])

            if self.time_int.subiter == 0:
                self.assertTrue(np.allclose(dsol, np.array([[0, 3e-7], [1e-7, 4e-7], [2e-7, 5e-7]])))
            elif self.time_int.subiter == 1:
                self.assertTrue(np.allclose(dsol, np.array([[2.5e-8, 1.75e-7], [7.5e-8, 2.25e-7], [1.25e-07, 2.75e-7]])))
            elif self.time_int.subiter == 2:
                self.assertTrue(np.allclose(dsol, np.array([[1.5e-7, 4.5e-7], [2.5e-7, 5.5e-7], [(17/6)*1e-7, 6.5e-7]])))

# TODO: Jameson low storage

import unittest

from perform.time_integrator.time_integrator import TimeIntegrator


class TimeIntegratorTestCase(unittest.TestCase):
    def setUp(self):

        # set up param_dict
        self.param_dict = {}
        self.param_dict["dt"] = 1e-7
        self.param_dict["time_scheme"] = "bdf"
        self.param_dict["time_order"] = 2

    def test_time_integrator_init(self):

        time_integrator = TimeIntegrator(self.param_dict)
        self.assertEqual(time_integrator.dt, 1e-7)
        self.assertEqual(time_integrator.time_scheme, "bdf")
        self.assertEqual(time_integrator.time_order, 2)
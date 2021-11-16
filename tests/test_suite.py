import unittest

import test_constants
import test_input_funcs
import test_misc_funcs
import test_mesh
import test_system_solver
from test_gas_model import test_gas_model, test_cpg
from test_time_integrator import test_time_integrator, test_explicit_integrator

from regression_tests.examples_fom import test_examples_fom

loader = unittest.TestLoader()

def init_unit_test_suite():
    """Test cases for initializing independent classes and unit testing their member methods"""

    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(test_constants.ConstantsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_input_funcs.InputParsersTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_misc_funcs.MiscFuncsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_mesh.MeshTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_system_solver.SystemSolverInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_gas_model.GasModelInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_gas_model.GasModelMethodsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_cpg.CPGInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_cpg.GasModelMethodsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_time_integrator.TimeIntegratorTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_explicit_integrator.ExplicitTimeIntInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_explicit_integrator.ClassicRK4InitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_explicit_integrator.SSPRK3InitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_explicit_integrator.ClassicRK4MethodsTestCase))
    return suite


def regression_test_suite():
    """Test cases for running full simulation examples"""

    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(test_examples_fom.FOMRegressionTests))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(init_unit_test_suite())
    runner.run(regression_test_suite())

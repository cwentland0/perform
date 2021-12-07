import unittest
import sys

import test_constants
import test_input_funcs
import test_misc_funcs
import test_mesh
import test_system_solver
from test_gas_model import test_gas_model, test_cpg
from test_time_integrator import test_time_integrator, test_explicit_integrator, test_implicit_integrator


loader = unittest.TestLoader()


def indep_unit_test_suite():
    """Test cases for independent class unit tests"""

    suite = unittest.TestSuite()

    suite.addTest(loader.loadTestsFromTestCase(test_constants.ConstantsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_input_funcs.InputParsersTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_misc_funcs.MiscFuncsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_mesh.MeshTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_system_solver.SystemSolverInitTestCase))

    return suite


def gas_model_test_suite():
    """Test cases for gas model unit tests"""

    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(test_gas_model.GasModelInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_gas_model.GasModelMethodsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_cpg.CPGInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_cpg.GasModelMethodsTestCase))

    return suite


def time_int_test_suite():
    """Test cases for time integrator unit tests"""

    suite = unittest.TestSuite()

    suite.addTest(loader.loadTestsFromTestCase(test_time_integrator.TimeIntegratorTestCase))

    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.ExplicitTimeIntInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.ClassicRK4InitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.SSPRK3InitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.JamesonLowStoreInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.ClassicRK4MethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.SSPRK3MethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_explicit_integrator.JamesonLowStoreMethodsTestCase))

    suite.addTest(loader.loadTestsFromTestCase(test_implicit_integrator.ImplicitTimeIntInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_implicit_integrator.BDFTestCase))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    ret_list = []
    ret_list.append(runner.run(indep_unit_test_suite()).wasSuccessful())
    ret_list.append(runner.run(gas_model_test_suite()).wasSuccessful())
    ret_list.append(runner.run(time_int_test_suite()).wasSuccessful())

    ret = not all(ret_list)
    sys.exit(ret)  # this is NOT the recommended usage, need to figure out how to use unittest.main()

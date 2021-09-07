import unittest

import test_constants
import test_input_funcs
import test_misc_funcs
import test_mesh
import test_system_solver
from test_gas_model import test_gas_model


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(test_constants.ConstantsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_input_funcs.InputParsersTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_misc_funcs.MiscFuncsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_mesh.MeshTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_system_solver.SystemSolverInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_gas_model.GasModelInitTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_gas_model.GasModelMethodsTestCase))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite())

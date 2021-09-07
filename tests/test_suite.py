import unittest

import test_constants
import test_input_funcs
import test_misc_funcs
import test_mesh


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(test_constants.ConstantsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_input_funcs.InputParsersTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_misc_funcs.MiscFuncsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(test_mesh.MeshTestCase))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite())

import unittest
import sys

from examples_fom import test_examples_fom


loader = unittest.TestLoader()


def regression_test_suite():
    """Test cases for running full simulation examples"""

    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(test_examples_fom.FOMRegressionTests))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    ret = not runner.run(regression_test_suite()).wasSuccessful()
    sys.exit(ret)  # this is NOT the recommended usage, need to figure out how to use unittest.main()

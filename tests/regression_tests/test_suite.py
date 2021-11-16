import unittest

from examples_fom import test_examples_fom


loader = unittest.TestLoader()


def regression_test_suite():
    """Test cases for running full simulation examples"""

    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(test_examples_fom.FOMRegressionTests))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(regression_test_suite())

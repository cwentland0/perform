import unittest

from test_solution import test_solution_phys

loader = unittest.TestLoader()


def solution_integration_test_suite():
    """Test cases for independent class unit tests"""

    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(test_solution_phys.SolutionPhysInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_solution_phys.SolutionPhysMethodsTestCase))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(solution_integration_test_suite())

import unittest

from test_solution import test_solution_phys, test_solution_interior

loader = unittest.TestLoader()


def solution_integration_test_suite():

    suite = unittest.TestSuite()

    # initializations
    suite.addTests(loader.loadTestsFromTestCase(test_solution_phys.SolutionPhysInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_solution_interior.SolutionIntInitTestCase))

    # methods
    suite.addTest(loader.loadTestsFromTestCase(test_solution_phys.SolutionPhysMethodsTestCase))
    

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(solution_integration_test_suite())

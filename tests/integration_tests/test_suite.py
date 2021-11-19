import unittest
import os
import shutil
import subprocess
from argparse import ArgumentParser

from test_solution import test_solution_phys, test_solution_interior
from test_reaction_model import test_finite_rate_irrev_reaction
from test_flux.invisc_flux import test_invisc_flux

loader = unittest.TestLoader()


def integration_test_suite(output_mode=False):

    suite = unittest.TestSuite()

    # initializations
    suite.addTests(loader.loadTestsFromTestCase(test_solution_phys.SolutionPhysInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_solution_interior.SolutionIntInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_finite_rate_irrev_reaction.FiniteRateIrrevReactionInitTestCase))

    # methods
    suite.addTest(loader.loadTestsFromTestCase(test_solution_phys.SolutionPhysMethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_solution_interior.SolutionIntMethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_finite_rate_irrev_reaction.FiniteRateIrrevReactionMethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_invisc_flux.InviscFluxMethodsTestCase))

    return suite


if __name__ == "__main__":

    # Check whether to run tests in output mode
    # In output mode, will not check method results, but will instead save them to disk
    # These outputs are retrieved from remote storage and used for comparison when not in output mode
    parser = ArgumentParser()
    parser.add_argument("-o", action="store_true", dest="output_mode", default=False, help="Run method tests in output mode")
    output_mode = parser.parse_args().output_mode
    
    # clear out the current output directory and remake it
    localdir = os.path.dirname(__file__)
    # handle weirdness with GHA and subprocess
    if localdir in ["", "."]:
        localdir = "./"
    outputdir = os.path.join(localdir, "output_dir")
    if os.path.isdir(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    os.environ["PERFORM_TEST_OUTPUT_DIR"] = outputdir

    if output_mode:
        os.environ["PERFORM_TEST_OUTPUT_MODE"] = "1"
    else:
        os.environ["PERFORM_TEST_OUTPUT_MODE"] = "0"
        # retrieve current "truth" results
        subprocess.call(os.path.join(localdir, "get_results.sh"))

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(integration_test_suite(output_mode=output_mode))


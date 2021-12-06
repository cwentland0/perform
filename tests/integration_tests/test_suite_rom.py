import unittest
import os
import shutil
import subprocess
from argparse import ArgumentParser
import sys

from test_rom.test_ml_library import test_tfkeras_library
from test_rom import test_rom_domain, test_rom_variable_mapping
from test_rom.test_rom_space_mapping import test_linear_space_mapping, test_autoencoder_space_mapping
from test_rom.test_rom_method.test_projection_method import test_galerkin_projection

loader = unittest.TestLoader()


def integration_test_suite():

    suite = unittest.TestSuite()

    # RomDomain initialization
    # NOTE: basically all ROM classes require RomDomain, so this has to come first
    suite.addTest(loader.loadTestsFromTestCase(test_rom_domain.RomDomainInitTestCase))

    # MLLibrary tests
    suite.addTest(loader.loadTestsFromTestCase(test_tfkeras_library.TFKerasLibraryMethodsTestCase))

    # RomVariableMapping tests
    suite.addTest(loader.loadTestsFromTestCase(test_rom_variable_mapping.RomPrimVarMappingMethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_rom_variable_mapping.RomConsVarMappingMethodsTestCase))

    # RomSpaceMapping tests
    suite.addTest(loader.loadTestsFromTestCase(test_linear_space_mapping.LinearSpaceMappingInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_linear_space_mapping.LinearSpaceMappingMethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_autoencoder_space_mapping.AutoencoderSpaceMappingInitTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_autoencoder_space_mapping.AutoencoderSpaceMappingMethodsTestCase))

    # RomMethod tests
    suite.addTest(loader.loadTestsFromTestCase(test_galerkin_projection.GalerkinProjectionLinearMethodsTestCase))
    suite.addTest(loader.loadTestsFromTestCase(test_galerkin_projection.GalerkinProjectionAutoencoderMethodsTestCase))

    return suite


if __name__ == "__main__":

    # Check whether to run tests in output mode
    # In output mode, will not check method results, but will instead save them to disk
    # These outputs are retrieved from remote storage and used for comparison when not in output mode
    parser = ArgumentParser()
    parser.add_argument(
        "-o", action="store_true", dest="output_mode", default=False, help="Run method tests in output mode"
    )
    output_mode = parser.parse_args().output_mode

    # clear out the current output directory and remake it
    localdir = os.path.dirname(__file__)
    # handle weirdness with GHA and subprocess
    if localdir in ["", "."]:
        localdir = "./"
    outputdir = os.path.join(localdir, "output_dir_rom")

    if os.path.isdir(outputdir):
        shutil.rmtree(outputdir)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    os.environ["PERFORM_TEST_OUTPUT_DIR"] = outputdir

    if output_mode:
        os.environ["PERFORM_TEST_OUTPUT_MODE"] = "1"
    else:
        os.environ["PERFORM_TEST_OUTPUT_MODE"] = "0"
        # retrieve current "truth" results
        subprocess.call(os.path.join(localdir, "get_results_rom.sh"))

    runner = unittest.TextTestRunner(verbosity=3)
    ret = not runner.run(integration_test_suite()).wasSuccessful()
    sys.exit(ret)  # this is NOT the recommended usage, need to figure out how to use unittest.main()

import unittest
import os
import shutil
import subprocess
from argparse import ArgumentParser
import sys

import numpy as np

from constants import solution_domain_setup
from perform.constants import UNSTEADY_OUTPUT_DIR_NAME, PROBE_OUTPUT_DIR_NAME, RESTART_OUTPUT_DIR_NAME
from perform.driver import driver


class DriverTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
        self.output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # generate input text files
        solution_domain_setup(self.test_dir)

        # generate initial condition file
        self.sol_prim_in = np.array(
            [
                [1e6, 9e5],
                [2.0, 1.0],
                [1000.0, 1200.0],
                [0.6, 0.4],
            ]
        )
        np.save(os.path.join(self.test_dir, "test_init_file.npy"), self.sol_prim_in)

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_driver(self):

        # run driver
        driver(self.test_dir)

        # ----- load output data from disk -----

        # unsteady outputs
        unst_out_dir = os.path.join(self.test_dir, UNSTEADY_OUTPUT_DIR_NAME)
        sol_prim = np.load(os.path.join(unst_out_dir, "sol_prim_FOM.npy"))
        sol_cons = np.load(os.path.join(unst_out_dir, "sol_cons_FOM.npy"))
        source = np.load(os.path.join(unst_out_dir, "source_FOM.npy"))
        heat_release = np.load(os.path.join(unst_out_dir, "heat_release_FOM.npy"))
        rhs = np.load(os.path.join(unst_out_dir, "rhs_FOM.npy"))

        # probe output
        probe_dir = os.path.join(self.test_dir, PROBE_OUTPUT_DIR_NAME)
        probe_1 = np.load(os.path.join(probe_dir, "probe_pressure_temperature_species-0_density_1_FOM.npy"))
        probe_2 = np.load(os.path.join(probe_dir, "probe_pressure_temperature_species-0_density_2_FOM.npy"))
        probe_3 = np.load(os.path.join(probe_dir, "probe_pressure_temperature_species-0_density_3_FOM.npy"))

        # restart file output
        restart_dir = os.path.join(self.test_dir, RESTART_OUTPUT_DIR_NAME)
        restart_1 = np.load(os.path.join(restart_dir, "restart_file_1.npz"))
        restart_2 = np.load(os.path.join(restart_dir, "restart_file_2.npz"))

        if self.output_mode:

            # unsteady outputs
            np.save(os.path.join(self.output_dir, "driver_sol_prim.npy"), sol_prim)
            np.save(os.path.join(self.output_dir, "driver_sol_cons.npy"), sol_cons)
            np.save(os.path.join(self.output_dir, "driver_source.npy"), source)
            np.save(os.path.join(self.output_dir, "driver_heat_release.npy"), heat_release)
            np.save(os.path.join(self.output_dir, "driver_rhs.npy"), rhs)

            # probes
            np.save(os.path.join(self.output_dir, "driver_probe_1.npy"), probe_1)
            np.save(os.path.join(self.output_dir, "driver_probe_2.npy"), probe_2)
            np.save(os.path.join(self.output_dir, "driver_probe_3.npy"), probe_3)

            # restart files
            np.save(os.path.join(self.output_dir, "driver_restart_1_prim.npy"), restart_1["sol_prim"])
            np.save(os.path.join(self.output_dir, "driver_restart_1_cons.npy"), restart_1["sol_cons"])
            np.save(os.path.join(self.output_dir, "driver_restart_2_prim.npy"), restart_2["sol_prim"])
            np.save(os.path.join(self.output_dir, "driver_restart_2_cons.npy"), restart_2["sol_cons"])

        else:

            # unsteady outputs
            self.assertTrue(np.array_equal(sol_prim, np.load(os.path.join(self.output_dir, "driver_sol_prim.npy"))))
            self.assertTrue(np.array_equal(sol_cons, np.load(os.path.join(self.output_dir, "driver_sol_cons.npy"))))
            self.assertTrue(np.array_equal(source, np.load(os.path.join(self.output_dir, "driver_source.npy"))))
            self.assertTrue(
                np.array_equal(heat_release, np.load(os.path.join(self.output_dir, "driver_heat_release.npy")))
            )
            self.assertTrue(np.array_equal(rhs, np.load(os.path.join(self.output_dir, "driver_rhs.npy"))))

            # probe data
            self.assertTrue(np.array_equal(probe_1, np.load(os.path.join(self.output_dir, "driver_probe_1.npy"))))
            self.assertTrue(np.array_equal(probe_2, np.load(os.path.join(self.output_dir, "driver_probe_2.npy"))))
            self.assertTrue(np.array_equal(probe_3, np.load(os.path.join(self.output_dir, "driver_probe_3.npy"))))

            # restart files
            self.assertTrue(
                np.array_equal(
                    restart_1["sol_prim"], np.load(os.path.join(self.output_dir, "driver_restart_1_prim.npy"))
                )
            )
            self.assertTrue(
                np.array_equal(
                    restart_1["sol_cons"], np.load(os.path.join(self.output_dir, "driver_restart_1_cons.npy"))
                )
            )
            self.assertTrue(
                np.array_equal(
                    restart_2["sol_prim"], np.load(os.path.join(self.output_dir, "driver_restart_2_prim.npy"))
                )
            )
            self.assertTrue(
                np.array_equal(
                    restart_2["sol_cons"], np.load(os.path.join(self.output_dir, "driver_restart_2_cons.npy"))
                )
            )


if __name__ == "__main__":

    # Check whether to run in output mode
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
    outputdir = os.path.join(localdir, "output_dir_driver")

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
        subprocess.call(os.path.join(localdir, "get_results_driver.sh"))

    runner = unittest.TextTestRunner(verbosity=3)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTest(loader.loadTestsFromTestCase(DriverTestCase))
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)  # this is NOT the recommended usage, need to figure out how to use unittest.main()

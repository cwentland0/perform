import unittest
import shutil
import os

from perform.system_solver import SystemSolver
import perform.constants as constants


class SystemSolverInitTestCase(unittest.TestCase):
    def setUp(self):

        # generate working directory
        self.test_dir = "test_dir"
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.mkdir(self.test_dir)

        # generate file with necessary input files
        self.test_file = os.path.join(self.test_dir, constants.PARAM_INPUTS)
        with open(self.test_file, "w") as f:
            f.write('init_file = "test_init_file.npy"\n')
            f.write("dt = 1e-8\n")
            f.write('time_scheme = "bdf"\n')
            f.write("num_steps = 100\n")
            f.write("save_restarts = True\n")

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_system_solver_init(self):

        # TODO: check iter, time_iter, restart_iter once the one/zero indexing thing is settled

        solver = SystemSolver(self.test_dir)

        # check creation of directories
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, constants.UNSTEADY_OUTPUT_DIR_NAME)))
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, constants.PROBE_OUTPUT_DIR_NAME)))
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, constants.IMAGE_OUTPUT_DIR_NAME)))
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, constants.RESTART_OUTPUT_DIR_NAME)))

        # check set inputs
        self.assertEqual(solver.init_file, os.path.join(self.test_dir, "test_init_file.npy"))
        self.assertEqual(solver.dt, 1e-8)
        self.assertEqual(solver.num_steps, 100)
        self.assertTrue(solver.save_restarts)

        # check required defaults
        self.assertFalse(solver.run_steady)
        self.assertEqual(solver.sol_time, 0.0)
        self.assertTrue(solver.num_restarts >= 1)
        self.assertTrue(solver.restart_interval >= 1)
        self.assertTrue(solver.out_interval >= 1)
        self.assertEqual(solver.vel_add, 0.0)
        self.assertFalse(solver.solve_failed)
        self.assertFalse(solver.source_off)
        self.assertFalse(solver.calc_rom)


if __name__ == "__main__":
    unittest.main()

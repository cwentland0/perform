import unittest
import os
import shutil

import numpy as np

from constants import solution_domain_setup, rom_domain_setup
from perform.constants import REAL_TYPE
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.rom.rom_domain import RomDomain


class RomDomainInitTestCase(unittest.TestCase):
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
        rom_domain_setup(self.test_dir)

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

        # generate spatial modes, standardization profiles
        self.model_dir = os.path.join(self.test_dir, "model_files")
        os.mkdir(self.model_dir)

        modes = np.reshape(np.eye(8), (4, 2, 8))
        cent_prof = np.zeros((4, 2), dtype=REAL_TYPE)
        norm_sub_prof = np.zeros((4, 2), dtype=REAL_TYPE)
        norm_fac_prof = np.ones((4, 2), dtype=REAL_TYPE)

        np.save(os.path.join(self.model_dir, "spatial_modes.npy"), modes)
        np.save(os.path.join(self.model_dir, "cent_prof_prim.npy"), cent_prof)
        np.save(os.path.join(self.model_dir, "norm_sub_prof_prim.npy"), norm_sub_prof)
        np.save(os.path.join(self.model_dir, "norm_fac_prof_prim.npy"), norm_fac_prof)
        np.save(os.path.join(self.model_dir, "norm_sub_prof_cons.npy"), norm_sub_prof)
        np.save(os.path.join(self.model_dir, "norm_fac_prof_cons.npy"), norm_fac_prof)

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(self.test_dir)
        self.sol_domain = SolutionDomain(self.solver)

        # set model_dir
        self.solver.rom_dict["model_dir"] = self.model_dir

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_rom_domain_init(self):

        rom_domain = RomDomain(self.sol_domain, self.solver)
        
        self.assertEqual(rom_domain.num_models, 1)
        self.assertTrue(np.array_equal(rom_domain.latent_dims, [8]))
        self.assertEqual(rom_domain.latent_dim_total, 8)
        self.assertTrue(np.array_equal(self.sol_domain.sol_int.sol_prim, self.sol_prim_in))

        # TODO: could probably check some other stuff
        
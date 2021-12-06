import unittest
import os

import numpy as np

from constants import (
    SOL_PRIM_IN_REACT,
    TEST_DIR,
    del_test_dir,
    gen_test_dir,
    solution_domain_setup,
    rom_domain_setup,
    get_output_mode,
)
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain
from perform.rom.rom_domain import RomDomain


class NumericalTimeStepperMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="galerkin", space_mapping="linear", var_mapping="conservative")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.solver.param_dict["dual_time"] = False
        self.sol_domain = SolutionDomain(self.solver)
        self.rom_domain = RomDomain(self.sol_domain, self.solver)
        self.time_stepper = self.rom_domain.time_stepper

    def tearDown(self):

        del_test_dir()

    # def test_init_state(self):

    #     pass

    def test_advance_subiter(self):

        self.time_stepper.time_integrator.subiter = 0
        self.time_stepper.advance_subiter(self.sol_domain, self.solver, self.rom_domain)

        model = self.rom_domain.model_list[0]
        if self.output_mode:
            np.save(
                os.path.join(self.output_dir, "numerical_stepper_subiter_sol_prim.npy"),
                self.sol_domain.sol_int.sol_prim,
            )
            np.save(
                os.path.join(self.output_dir, "numerical_stepper_subiter_sol_cons.npy"),
                self.sol_domain.sol_int.sol_cons,
            )
            np.save(os.path.join(self.output_dir, "numerical_stepper_subiter_res.npy"), model.res)
            np.save(os.path.join(self.output_dir, "numerical_stepper_subiter_d_code.npy"), model.d_code)
            np.save(os.path.join(self.output_dir, "numerical_stepper_subiter_code.npy"), model.code)

        else:
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_prim,
                    np.load(os.path.join(self.output_dir, "numerical_stepper_subiter_sol_prim.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_cons,
                    np.load(os.path.join(self.output_dir, "numerical_stepper_subiter_sol_cons.npy")),
                )
            )
            self.assertTrue(
                np.allclose(model.res, np.load(os.path.join(self.output_dir, "numerical_stepper_subiter_res.npy")))
            )
            self.assertTrue(
                np.allclose(
                    model.d_code, np.load(os.path.join(self.output_dir, "numerical_stepper_subiter_d_code.npy"))
                )
            )
            self.assertTrue(
                np.allclose(model.code, np.load(os.path.join(self.output_dir, "numerical_stepper_subiter_code.npy")))
            )

    def test_advance_iter(self):

        self.time_stepper.advance_iter(self.sol_domain, self.solver, self.rom_domain)

        model = self.rom_domain.model_list[0]
        if self.output_mode:
            np.save(
                os.path.join(self.output_dir, "numerical_stepper_iter_sol_prim.npy"), self.sol_domain.sol_int.sol_prim
            )
            np.save(
                os.path.join(self.output_dir, "numerical_stepper_iter_sol_cons.npy"), self.sol_domain.sol_int.sol_cons
            )
            np.save(os.path.join(self.output_dir, "numerical_stepper_iter_res.npy"), model.res)
            np.save(os.path.join(self.output_dir, "numerical_stepper_iter_d_code.npy"), model.d_code)
            np.save(os.path.join(self.output_dir, "numerical_stepper_iter_code.npy"), model.code)

        else:
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_prim,
                    np.load(os.path.join(self.output_dir, "numerical_stepper_iter_sol_prim.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_cons,
                    np.load(os.path.join(self.output_dir, "numerical_stepper_iter_sol_cons.npy")),
                )
            )
            self.assertTrue(
                np.allclose(model.res, np.load(os.path.join(self.output_dir, "numerical_stepper_iter_res.npy")))
            )
            self.assertTrue(
                np.allclose(model.d_code, np.load(os.path.join(self.output_dir, "numerical_stepper_iter_d_code.npy")))
            )
            self.assertTrue(
                np.allclose(model.code, np.load(os.path.join(self.output_dir, "numerical_stepper_iter_code.npy")))
            )

    def test_calc_code_res_norms(self):

        self.solver.iter = 3
        self.rom_domain.model_list[0].res[:] = SOL_PRIM_IN_REACT.ravel(order="C")

        self.time_stepper.calc_code_res_norms(self.sol_domain, self.solver, self.rom_domain)

        self.assertAlmostEqual(self.sol_domain.sol_int.res_norm_l2, 475657.76037051)
        self.assertAlmostEqual(self.sol_domain.sol_int.res_norm_l1, 237775.5)
        self.assertAlmostEqual(self.sol_domain.sol_int.res_norm_hist[2, 0], 475657.76037051)
        self.assertAlmostEqual(self.sol_domain.sol_int.res_norm_hist[2, 1], 237775.5)

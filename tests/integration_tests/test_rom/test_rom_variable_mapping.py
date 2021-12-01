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


class RomPrimVarMappingMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()
        rom_domain_setup(method="mplsvt", space_mapping="linear", var_mapping="primitive")

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)
        self.rom_domain = RomDomain(self.sol_domain, self.solver)
        self.var_mapping = self.rom_domain.var_mapping

    def tearDown(self):

        del_test_dir()

    def test_prim_var_map_get_variables(self):

        # check get_variables_from_state
        sol_prim = self.var_mapping.get_variables_from_state(self.sol_domain)
        self.assertTrue(np.allclose(sol_prim, SOL_PRIM_IN_REACT))

        # check get_variable_hist_from_state_hist
        sol_prim_hist = self.var_mapping.get_variable_hist_from_state_hist(self.sol_domain)
        self.assertTrue(all([np.allclose(sol_prim_arr, SOL_PRIM_IN_REACT) for sol_prim_arr in sol_prim_hist]))

    def test_prim_var_map_update_state(self):

        sol_cons_old = self.sol_domain.sol_int.sol_cons.copy()

        # make "new" state
        sol_prim_new = SOL_PRIM_IN_REACT.copy()
        sol_prim_new[0, :] += 1000.0
        sol_prim_new[1, :] += 1.0
        sol_prim_new[2, :] += 100.0

        # update RomModel internal state
        rom_model = self.rom_domain.model_list[0]
        rom_model.sol[:, :] = sol_prim_new.copy()
        rom_model.sol_hist[0][:, :] = sol_prim_new.copy()

        # check update_full_state
        self.var_mapping.update_full_state(self.sol_domain, self.rom_domain)

        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_prim, sol_prim_new))

        if self.output_mode:
            np.save(
                os.path.join(self.output_dir, "prim_var_map_update_full_sol_cons.npy"), self.sol_domain.sol_int.sol_cons
            )

        else:
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_cons,
                    np.load(os.path.join(self.output_dir, "prim_var_map_update_full_sol_cons.npy")),
                )
            )

        # check update_state_hist
        self.var_mapping.update_state_hist(self.sol_domain, self.rom_domain)

        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_prim[0], sol_prim_new))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_prim[1], SOL_PRIM_IN_REACT))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_prim[2], SOL_PRIM_IN_REACT))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_cons[0], self.sol_domain.sol_int.sol_cons))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_cons[1], sol_cons_old))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_cons[2], sol_cons_old))


class RomConsVarMappingMethodsTestCase(unittest.TestCase):
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
        self.var_mapping = self.rom_domain.var_mapping

    def tearDown(self):

        del_test_dir()

    def test_cons_var_map_get_variables(self):

        # check get_variables_from_state
        sol_cons = self.var_mapping.get_variables_from_state(self.sol_domain)
        self.assertTrue(np.allclose(sol_cons, self.sol_domain.sol_int.sol_cons))

        # check get_variable_hist_from_state_hist
        sol_cons_hist = self.var_mapping.get_variable_hist_from_state_hist(self.sol_domain)
        self.assertTrue(
            all([np.allclose(sol_cons_arr, self.sol_domain.sol_int.sol_cons) for sol_cons_arr in sol_cons_hist])
        )

    def test_cons_var_map_update_state(self):

        sol_cons_old = self.sol_domain.sol_int.sol_cons.copy()

        # make "new" state
        sol_cons_new = sol_cons_old.copy()
        sol_cons_new[0, :] += 0.25
        sol_cons_new[1, :] += 1.0
        sol_cons_new[2, :] += 2.5e6

        # update RomModel internal state
        rom_model = self.rom_domain.model_list[0]
        rom_model.sol[:, :] = sol_cons_new.copy()
        rom_model.sol_hist[0][:, :] = sol_cons_new.copy()

        # check update_full_state
        self.var_mapping.update_full_state(self.sol_domain, self.rom_domain)

        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_cons, sol_cons_new))

        if self.output_mode:
            np.save(
                os.path.join(self.output_dir, "cons_var_map_update_full_sol_prim.npy"), self.sol_domain.sol_int.sol_prim
            )

        else:
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_prim,
                    np.load(os.path.join(self.output_dir, "cons_var_map_update_full_sol_prim.npy")),
                )
            )

        # check update_state_hist
        self.var_mapping.update_state_hist(self.sol_domain, self.rom_domain)

        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_prim[0], self.sol_domain.sol_int.sol_prim))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_prim[1], SOL_PRIM_IN_REACT))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_prim[2], SOL_PRIM_IN_REACT))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_cons[0], sol_cons_new))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_cons[1], sol_cons_old))
        self.assertTrue(np.allclose(self.sol_domain.sol_int.sol_hist_cons[2], sol_cons_old))

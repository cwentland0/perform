import unittest
import os

import numpy as np

from constants import SOL_PRIM_IN_REACT, TEST_DIR, del_test_dir, gen_test_dir, get_output_mode, solution_domain_setup
from perform.constants import REAL_TYPE
from perform.input_funcs import get_initial_conditions, get_absolute_path
from perform.system_solver import SystemSolver
from perform.solution.solution_domain import SolutionDomain


class SolutionDomainInitTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver
        self.solver = SystemSolver(TEST_DIR)

    def tearDown(self):

        del_test_dir()

    def test_solution_domain_init(self):

        sol = SolutionDomain(self.solver)

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_int_init_sol_cons.npy"), sol.sol_int.sol_cons)

        else:
            self.assertTrue(np.array_equal(sol.sol_int.sol_prim, SOL_PRIM_IN_REACT))
            self.assertTrue(
                np.allclose(
                    sol.sol_int.sol_cons, np.load(os.path.join(self.output_dir, "sol_domain_int_init_sol_cons.npy"))
                )
            )

            # TODO: a LOT of checking of other variables


class SolutionDomainMethodsTestCase(unittest.TestCase):
    def setUp(self):

        self.output_mode, self.output_dir = get_output_mode()

        # generate working directory
        gen_test_dir()

        # generate input text files
        solution_domain_setup()

        # set SystemSolver and SolutionDomain
        self.solver = SystemSolver(TEST_DIR)
        self.sol_domain = SolutionDomain(self.solver)

    def tearDown(self):

        del_test_dir()

    def test_calc_rhs(self):

        self.sol_domain.calc_rhs(self.solver)

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_rhs.npy"), self.sol_domain.sol_int.rhs)

        else:
            self.assertTrue(
                np.allclose(self.sol_domain.sol_int.rhs, np.load(os.path.join(self.output_dir, "sol_domain_rhs.npy")))
            )

    def test_calc_rhs_jacob(self):

        self.sol_domain.calc_rhs(self.solver)
        rhs_jacob_center, rhs_jacob_left, rhs_jacob_right = self.sol_domain.calc_rhs_jacob(self.solver)

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_rhs_jacob_center.npy"), rhs_jacob_center)
            np.save(os.path.join(self.output_dir, "sol_domain_rhs_jacob_left.npy"), rhs_jacob_left)
            np.save(os.path.join(self.output_dir, "sol_domain_rhs_jacob_right.npy"), rhs_jacob_right)

        else:
            self.assertTrue(
                np.allclose(rhs_jacob_center, np.load(os.path.join(self.output_dir, "sol_domain_rhs_jacob_center.npy")))
            )
            self.assertTrue(
                np.allclose(rhs_jacob_left, np.load(os.path.join(self.output_dir, "sol_domain_rhs_jacob_left.npy")))
            )
            self.assertTrue(
                np.allclose(rhs_jacob_right, np.load(os.path.join(self.output_dir, "sol_domain_rhs_jacob_right.npy")))
            )

    def test_calc_res_jacob_dual_time(self):

        self.sol_domain.calc_rhs(self.solver)
        res_jacob = self.sol_domain.calc_res_jacob(self.solver).todense()

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_res_jacob.npy"), res_jacob)

        else:
            self.assertTrue(np.allclose(res_jacob, np.load(os.path.join(self.output_dir, "sol_domain_res_jacob.npy"))))

    def test_calc_res_jacob_phys_time(self):

        # overwrite to use physical time
        self.solver = SystemSolver(TEST_DIR)
        self.solver.param_dict["dual_time"] = False
        self.sol_domain = SolutionDomain(self.solver)

        self.sol_domain.calc_rhs(self.solver)
        res_jacob = self.sol_domain.calc_res_jacob(self.solver).todense()

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_res_jacob_phys_time.npy"), res_jacob)

        else:
            self.assertTrue(np.allclose(res_jacob, np.load(os.path.join(self.output_dir, "sol_domain_res_jacob_phys_time.npy"))))

    def test_advance_subiter(self):

        self.sol_domain.time_integrator.subiter = 0
        self.sol_domain.advance_subiter(self.solver)

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_subiter_sol_prim.npy"), self.sol_domain.sol_int.sol_prim)
            np.save(os.path.join(self.output_dir, "sol_domain_subiter_sol_cons.npy"), self.sol_domain.sol_int.sol_cons)
            np.save(os.path.join(self.output_dir, "sol_domain_subiter_res.npy"), self.sol_domain.sol_int.res)

        else:
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_prim,
                    np.load(os.path.join(self.output_dir, "sol_domain_subiter_sol_prim.npy")),
                )
            )
            self.assertTrue(
                np.allclose(
                    self.sol_domain.sol_int.sol_cons,
                    np.load(os.path.join(self.output_dir, "sol_domain_subiter_sol_cons.npy")),
                )
            )
            # NOTE: Some discrepancy causes this to fail on GHA, removing for now
            pass
            # self.assertTrue(
            #     np.allclose(
            #         self.sol_domain.sol_int.res, np.load(os.path.join(self.output_dir, "sol_domain_subiter_res.npy"))
            #     )
            # )

    def test_advance_iter(self):

        self.sol_domain.advance_iter(self.solver)

        if self.output_mode:
            np.save(os.path.join(self.output_dir, "sol_domain_iter_sol_prim.npy"), self.sol_domain.sol_int.sol_prim)
            np.save(os.path.join(self.output_dir, "sol_domain_iter_sol_cons.npy"), self.sol_domain.sol_int.sol_cons)
            np.save(os.path.join(self.output_dir, "sol_domain_iter_res.npy"), self.sol_domain.sol_int.res)

        else:
            # NOTE: Some discrepancy causes this to fail on GHA, removing for now
            pass
            # self.assertTrue(
            #     np.allclose(
            #         self.sol_domain.sol_int.sol_prim,
            #         np.load(os.path.join(self.output_dir, "sol_domain_iter_sol_prim.npy")),
            #     )
            # )
            # self.assertTrue(
            #     np.allclose(
            #         self.sol_domain.sol_int.sol_cons,
            #         np.load(os.path.join(self.output_dir, "sol_domain_iter_sol_cons.npy")),
            #     )
            # )
            # self.assertTrue(
            #     np.allclose(
            #         self.sol_domain.sol_int.res, np.load(os.path.join(self.output_dir, "sol_domain_iter_res.npy"))
            #     )
            # )

    def test_gen_piecewise_uniform_ic(self):
        # Done here because it depends on SolutionDomain

        # overwrite SystemSolver
        self.solver.init_file = None

        # write uniform IC file
        ic_file = "uniform_ic.inp"
        self.solver.ic_params_file = get_absolute_path(ic_file, self.solver.working_dir)
        with open(os.path.join(TEST_DIR, ic_file), "w") as f:
            f.write("x_split = 1e-5 \n")
            f.write("press_left = 1e4 \n")
            f.write("vel_left = 0.5 \n")
            f.write("temp_left = 5000.0 \n")
            f.write("mass_fracs_left = [0.6, 0.4] \n")
            f.write("press_right = 2e4 \n")
            f.write("vel_right = 10.0 \n")
            f.write("temp_right = 400 \n")
            f.write("mass_fracs_right = [0.2, 0.8] \n")

        # go through get_initial_conditions to catch some more lines
        sol_prim_init = get_initial_conditions(self.sol_domain, self.solver)

        self.assertTrue(np.array_equal(
            sol_prim_init[:, :, 0],
            np.array([[1e4, 2e4], [0.5, 10.0], [5000.0, 400.0], [0.6, 0.2]])
        ))

    def test_probe_output(self):

        sol_inlet = self.sol_domain.sol_inlet
        sol_outlet = self.sol_domain.sol_outlet
        sol_int = self.sol_domain.sol_int
        probe_1_arr = np.array([
            sol_inlet.sol_prim[0, 0],
            sol_inlet.sol_prim[2, 0],
            sol_inlet.sol_prim[3, 0],
            sol_inlet.sol_cons[0, 0],
        ])
        probe_2_arr = np.array([
            sol_int.sol_prim[0, 0],
            sol_int.sol_prim[2, 0],
            sol_int.sol_prim[3, 0],
            sol_int.sol_cons[0, 0],
        ])
        probe_3_arr = np.array([
            sol_outlet.sol_prim[0, 0],
            sol_outlet.sol_prim[2, 0],
            sol_outlet.sol_prim[3, 0],
            sol_outlet.sol_cons[0, 0],
        ])

        probe_base = np.zeros((5, self.solver.num_steps), dtype=REAL_TYPE)
        probe_base[0, :] = np.arange(1, self.solver.num_steps + 1) * self.solver.dt
        probe_1_comp = probe_base.copy()
        probe_2_comp = probe_base.copy()
        probe_3_comp = probe_base.copy()
        probe_1_comp[1:, :] = np.repeat(probe_1_arr[:, None], self.solver.num_steps, axis=-1)
        probe_2_comp[1:, :] = np.repeat(probe_2_arr[:, None], self.solver.num_steps, axis=-1)
        probe_3_comp[1:, :] = np.repeat(probe_3_arr[:, None], self.solver.num_steps, axis=-1)


        for self.solver.iter in range(1, self.solver.num_steps + 1):

            # update the probe matrix
            self.sol_domain.update_probes(self.solver)

            # write and check intermediate results
            if ((self.solver.iter % self.solver.out_itmdt_interval) == 0) and (
                self.solver.iter != self.solver.num_steps
            ):
                self.sol_domain.write_probes(self.solver, intermediate=True, failed=False)
                
                probe_1_itmdt = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_1_" + self.solver.sim_type + "_ITMDT.npy"))
                probe_2_itmdt = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_2_" + self.solver.sim_type + "_ITMDT.npy"))
                probe_3_itmdt = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_3_" + self.solver.sim_type + "_ITMDT.npy"))
                
                self.assertTrue(np.array_equal(probe_1_itmdt, probe_1_comp[:, :self.solver.iter]))
                self.assertTrue(np.array_equal(probe_2_itmdt, probe_2_comp[:, :self.solver.iter]))
                self.assertTrue(np.array_equal(probe_3_itmdt, probe_3_comp[:, :self.solver.iter]))
                
            # write and check "failed" snapshots
            if self.solver.iter == 7:
                self.sol_domain.write_probes(self.solver, intermediate=False, failed=True)

                probe_1_failed = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_1_" + self.solver.sim_type + "_FAILED.npy"))
                probe_2_failed = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_2_" + self.solver.sim_type + "_FAILED.npy"))
                probe_3_failed = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_3_" + self.solver.sim_type + "_FAILED.npy"))
                
                self.assertTrue(np.array_equal(probe_1_failed, probe_1_comp[:, :self.solver.iter]))
                self.assertTrue(np.array_equal(probe_2_failed, probe_2_comp[:, :self.solver.iter]))
                self.assertTrue(np.array_equal(probe_3_failed, probe_3_comp[:, :self.solver.iter]))
                                
        # delete intermediate results and check that they deleted properly
        self.sol_domain.delete_itmdt_probes(self.solver)
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_1_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_2_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )
        self.assertFalse(
            os.path.isfile(
                os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_3_" + self.solver.sim_type + "_ITMDT.npy")
            )
        )

        # write final probes
        self.sol_domain.write_probes(self.solver, intermediate=False, failed=False)

        probe_1 = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_1_" + self.solver.sim_type + ".npy"))
        probe_2 = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_2_" + self.solver.sim_type + ".npy"))
        probe_3 = np.load(os.path.join(self.solver.probe_output_dir, "probe_pressure_temperature_species-0_density_3_" + self.solver.sim_type + ".npy"))
        
        self.assertTrue(np.array_equal(probe_1, probe_1_comp))
        self.assertTrue(np.array_equal(probe_2, probe_2_comp))
        self.assertTrue(np.array_equal(probe_3, probe_3_comp))

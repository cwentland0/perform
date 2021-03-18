import os
from math import floor, log

import numpy as np

import perform.constants as const
from perform.input_funcs import read_input_file, catch_input, catch_list
from perform.mesh import Mesh
from perform.misc_funcs import mkdir_shallow


class SystemSolver:
    """
    Container class for solver parameters
    """

    def __init__(self, working_dir):

        # input parameters from solverParams.inp
        self.working_dir = working_dir
        param_file = os.path.join(self.working_dir, const.PARAM_INPUTS)
        param_dict = read_input_file(param_file)
        self.param_dict = param_dict

        # Make output directories
        self.unsteady_output_dir = mkdir_shallow(
            self.working_dir,
            const.UNSTEADY_OUTPUT_DIR_NAME)
        self.probe_output_dir = mkdir_shallow(
            self.working_dir,
            const.PROBE_OUTPUT_DIR_NAME)
        self.image_output_dir = mkdir_shallow(
            self.working_dir,
            const.IMAGE_OUTPUT_DIR_NAME)
        self.restart_output_dir = mkdir_shallow(
            self.working_dir,
            const.RESTART_OUTPUT_DIR_NAME)

        # initial condition file
        try:
            self.init_file = str(param_dict["init_file"])
        except KeyError:
            self.init_file = None

        # temporal discretization
        self.dt = float(param_dict["dt"])
        self.time_scheme = str(param_dict["time_scheme"])
        self.run_steady = catch_input(param_dict, "run_steady", False)
        self.num_steps = int(param_dict["num_steps"])
        self.iter = 1
        self.sol_time = 0.0
        self.time_iter = 1

        if self.run_steady:
            self.steady_tol = catch_input(param_dict,
                                          "steady_tol",
                                          const.L2_STEADY_TOL_DEFAULT)

        # restart files
        # TODO: could move this to solutionDomain, not terribly necessary
        self.save_restarts = catch_input(param_dict, "save_restarts", False)
        if self.save_restarts:
            self.restart_interval = catch_input(param_dict,
                                                "restart_interval",
                                                100)
            self.num_restarts = catch_input(param_dict, "num_restarts", 20)
            self.restart_iter = 1
        self.init_from_restart = catch_input(param_dict,
                                             "init_from_restart",
                                             False)

        if (self.init_file is None) and (not self.init_from_restart):
            self.ic_params_file = str(param_dict["ic_params_file"])

        # unsteady output
        self.out_interval = catch_input(param_dict, "out_interval", 1)
        self.prim_out = catch_input(param_dict, "prim_out", True)
        self.cons_out = catch_input(param_dict, "cons_out", False)
        self.source_out = catch_input(param_dict, "source_out", False)
        self.rhs_out = catch_input(param_dict, "rhs_out", False)

        assert (self.out_interval > 0), (
            "out_interval must be a positive integer")
        self.num_snaps = int(self.num_steps / self.out_interval)

        # misc
        self.vel_add = catch_input(param_dict, "vel_add", 0.0)
        self.res_norm_prim = catch_input(param_dict, "res_norm_prim", [None])
        self.source_off = catch_input(param_dict, "source_off", False)
        self.solve_failed = False

        # visualization
        self.num_probes = 0
        self.probe_vars = []

        # ROM flag
        self.calc_rom = catch_input(param_dict, "calc_rom", False)
        if not self.calc_rom:
            self.sim_type = "FOM"
        else:
            self.sim_type = "ROM"
            self.rom_inputs = os.path.join(self.working_dir, const.ROM_INPUTS)

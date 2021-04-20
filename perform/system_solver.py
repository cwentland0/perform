import os
from math import floor, log

import numpy as np

import perform.constants as const
from perform.input_funcs import read_input_file, catch_input, catch_list
from perform.mesh import Mesh
from perform.misc_funcs import mkdir_shallow


class SystemSolver:
    """Container class for global solver parameters.

    This class simply stores parameters which apply to the entire simulation, across all SolutionDomain's.
    It handles much of the input from the solver parameter input file, as well as handling parameters for outputs.

    Args:
        working_dir: Working directory of the current simulation.

    Attributes:
        working_dir: Working directory of the current simulation.
        param_dict: Dictionary of parameters read from the solver parameters input file.
        unsteady_output_dir: Path to directory for unsteady field data output.
        probe_output_dir: Path to directory for probe monitor data output.
        image_output_dir: Path to directory for visualization plot image output.
        restart_output_dir: Path to directory for restart file output.
        init_file: Path to NumPy binary file of initial condition primitive solution profiles, if desired. 
        dt: Physical time step size, in seconds.
        time_scheme:
            String name of the numerical time integration scheme to apply to the SolutionDomain.
        run_steady: Boolean flag indicating whether to run in "steady" mode.
        num_steps: Total number of physical time steps to simulate.
        iter: One-indexed simulation time step iteration number, always starts from one.
        sol_time: Physical solution time of current time step iteration, in seconds.
        time_iter:
            One-indexed physical time step iteration. Currently starts from one, but at some point will be
            changed to allow different numbers when initializing from init_file or a restart file.
        steady_tol:
            Primitive solution change norm residual threshold below which the "steady" solve is
            considered to be "converged".
        save_restarts: Boolean flag indicating whether to save restart files.
        restart_interval: Time step iteration interval at which to save restart files, if save_restarts is True.
        num_restarts: Maximum number of restart files to retain.
        restart_iter: One-indexed current restart file number. Overwritten if initializing from a restart file.
        init_from_restart:
            Boolean flag indicating whether to load the primitive solution profile initial condition
            from a restart file.
        ic_params_file:
            Path to text file including input parameters for defining a piecewise uniform
            initial condition primitive solution profile.
        out_interval: Time step iteration interval at which to store unsteady field profile data in a snapshot matrix.
        prim_out: Boolean flag indicating whether to store and save primitive solution profile snapshots.
        cons_out: Boolean flag indicating whether to store and save conservative solution profile snapshots.
        source_out: Boolean flag indicating whether to store and save source term profile snapshots.
        rhs_out:
            Boolean flag indicating whether to store and save semi-discrete right-hand side function profile snapshots.
        num_snaps: Total number of snapshots to be stored and saved, assuming the simulation runs to completion.
        vel_add: Velocity to add to the entire initial condition velocity field, in m/s.
        res_norm_prim:
            List of normalization factors for primitive solution variables when computing
            linear solve residual or solution change norms.
        source_off: Boolean flag indicating whether the source term should be turned off.
        solve_failed: Boolean flag indicating whether the solution has failed.
        num_probes: Number of probe monitors to be recorded.
        probe_vars: List of strings for variables which the probe monitors are to measure.
        calc_rom: Boolean flag indicating whether to run a ROM simulation.
        sim_type: Either "FOM" for a FOM simulation, or "ROM" for a ROM simulation.
        rom_inputs: Path to ROM parameters input file.
    """

    # TODO: time_scheme should not be associated with SystemSolver


    def __init__(self, working_dir):

        # input parameters from solverParams.inp
        self.working_dir = working_dir
        param_file = os.path.join(self.working_dir, const.PARAM_INPUTS)
        param_dict = read_input_file(param_file)
        self.param_dict = param_dict

        # Make output directories
        self.unsteady_output_dir = mkdir_shallow(self.working_dir, const.UNSTEADY_OUTPUT_DIR_NAME)
        self.probe_output_dir = mkdir_shallow(self.working_dir, const.PROBE_OUTPUT_DIR_NAME)
        self.image_output_dir = mkdir_shallow(self.working_dir, const.IMAGE_OUTPUT_DIR_NAME)
        self.restart_output_dir = mkdir_shallow(self.working_dir, const.RESTART_OUTPUT_DIR_NAME)

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
            self.steady_tol = catch_input(param_dict, "steady_tol", const.L2_STEADY_TOL_DEFAULT)

        # restart files
        # TODO: could move this to solutionDomain, not terribly necessary
        self.save_restarts = catch_input(param_dict, "save_restarts", False)
        if self.save_restarts:
            self.restart_interval = catch_input(param_dict, "restart_interval", 100)
            self.num_restarts = catch_input(param_dict, "num_restarts", 20)
            self.restart_iter = 1
        self.init_from_restart = catch_input(param_dict, "init_from_restart", False)

        if (self.init_file is None) and (not self.init_from_restart):
            self.ic_params_file = str(param_dict["ic_params_file"])

        # unsteady output
        self.out_interval = catch_input(param_dict, "out_interval", 1)
        self.prim_out = catch_input(param_dict, "prim_out", True)
        self.cons_out = catch_input(param_dict, "cons_out", False)
        self.source_out = catch_input(param_dict, "source_out", False)
        self.rhs_out = catch_input(param_dict, "rhs_out", False)

        assert self.out_interval > 0, "out_interval must be a positive integer"
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

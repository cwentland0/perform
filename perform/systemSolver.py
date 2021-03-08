import os
from math import floor, log

import numpy as np

import perform.constants as const
from perform.inputFuncs import read_input_file, catch_input, catch_list
from perform.mesh import Mesh


class SystemSolver:
	"""
	Container class for input parameters, domain geometry, spatial/temporal discretization, and gas model
	"""

	def __init__(self):

		# input parameters from solverParams.inp
		param_file = os.path.join(const.working_dir, const.PARAM_INPUTS)
		param_dict = read_input_file(param_file)
		self.param_dict = param_dict

		# spatial domain
		mesh_file 	= str(param_dict["mesh_file"]) 		# mesh properties file (string)
		mesh_dict 	= read_input_file(mesh_file)
		self.mesh 	= Mesh(mesh_dict)
		# TODO: selection for different meshes, when implemented

		# initial condition file
		try:
			self.init_file	= str(param_dict["init_file"]) 
		except:
			self.init_file 	= None

		# temporal discretization
		self.dt 			= float(param_dict["dt"])		# physical time step
		self.time_scheme 	= str(param_dict["time_scheme"])
		self.run_steady 		= catch_input(param_dict, "run_steady", False) # run "steady" simulation
		self.num_steps 		= int(param_dict["num_steps"])	# total number of physical time iterations
		self.iter 			= 1 							# iteration number for current run
		self.sol_time 		= 0.0							# physical time
		self.time_iter 		= 1 							# physical time iteration number

		if self.run_steady:
			self.steady_tol = catch_input(param_dict, "steady_tol", const.l2SteadyTolDefault) # threshold on convergence
		
		# spatial discretization parameters
		self.space_scheme 	= catch_input(param_dict, "space_scheme", "roe")	# spatial discretization scheme (string)
		self.space_order 	= catch_input(param_dict, "space_order", 1)		# spatial discretization order of accuracy (int)
		self.grad_limiter 	= catch_input(param_dict, "grad_limiter", "")		# gradient limiter for higher-order face reconstructions
		self.visc_scheme 	= catch_input(param_dict, "visc_scheme", 0)		# 0 for inviscid, 1 for viscous

		# restart files
		# TODO: could move this to solutionDomain, not terribly necessary
		self.save_restarts 	= catch_input(param_dict, "save_restarts", False) 	# whether to save restart files
		if self.save_restarts:
			self.restart_interval 	= catch_input(param_dict, "restart_interval", 100)	# number of steps between restart file saves
			self.num_restarts 		= catch_input(param_dict, "num_restarts", 20) 		# number of restart files to keep saved
			self.restart_iter 		= 1												# file number counter
		self.init_from_restart = catch_input(param_dict, "init_from_restart", False)

		if ((self.init_file == None) and (not self.init_from_restart)):
			try:
				self.ic_params_file 	= str(param_dict["ic_params_file"])
			except:
				raise KeyError("If not providing IC profile or restarting from restart file, must provide ic_params_file")

		# unsteady output
		self.out_interval   = catch_input(param_dict, "out_interval", 1) 		# iteration interval to save data (int)
		self.prim_out       = catch_input(param_dict, "prim_out", True)		# whether to save the primitive variables
		self.cons_out       = catch_input(param_dict, "cons_out", False) 		# whether to save the conservative variables
		self.source_out     = catch_input(param_dict, "source_out", False) 	# whether to save the species source term
		self.rhs_out        = catch_input(param_dict, "rhs_out", False)		# whether to save the RHS vector

		assert (self.out_interval > 0), "out_interval must be a positive integer"
		self.num_snaps 		= int(self.num_steps / self.out_interval)

		# misc
		self.vel_add        = catch_input(param_dict, "vel_add", 0.0)
		self.res_norm_prim  = catch_input(param_dict, "res_norm_prim", [None])
		self.source_on      = catch_input(param_dict, "source_on", True)
		self.solve_failed   = False

		# visualization
		self.num_probes = 0
		self.probe_vars = []

		# ROM flag
		self.calc_rom = catch_input(param_dict, "calc_rom", False)
		if not self.calc_rom: 
			self.sim_type = "FOM"
		else:
			self.sim_type = "ROM"
			self.romInputs = os.path.join(const.working_dir, const.ROM_INPUTS)
import os
import re

import numpy as np

import perform.constants as const


def catch_input(in_dict, in_key, default_val):
	"""
	Assign default values if user does not provide a certain input
	"""

	# TODO: correct error handling if default type is not recognized
	# TODO: check against lowercase'd strings so that inputs are not case sensitive. Do this for True/False too
	# TODO: instead of trusting user for NoneType, could also use NaN/Inf to indicate int/float defaults without passing a numerical default
	# 		or could just pass the actual default type lol, that'd be easier

	default_type = type(default_val)
	try:
		# if NoneType passed as default, trust user
		if (default_type == type(None)):
			outVal = in_dict[in_key]
		else:
			outVal = default_type(in_dict[in_key])
	except:
		outVal = default_val

	return outVal


def catch_list(in_dict, in_key, default, lenHighest=1):
	"""
	Input processor for reading lists or lists of lists
	Default defines length of lists at lowest level
	"""

	# TODO: needs to throw an error if input list of lists is longer than lenHighest
	# TODO: could make a recursive function probably, just hard to define appropriate list lengths at each level

	list_of_lists_flag = (type(default[0]) == list)
	
	try:
		inList = in_dict[in_key]

		if (len(inList) == 0):
			raise ValueError

		# list of lists
		if list_of_lists_flag:
			typeDefault = type(default[0][0])
			valList = []
			for listIdx in range(lenHighest):
				# if default type is NoneType, trust user
				if (typeDefault == type(None)):
					valList.append(inList[listIdx])
				else:
					castInList = [typeDefault(inVal) for inVal in inList[listIdx]]
					valList.append(castInList)

		# normal list
		else:
			typeDefault = type(default[0])
			# if default type is NoneType, trust user 
			if (typeDefault == type(None)):
				valList = inList
			else:
				valList = [typeDefault(inVal) for inVal in inList]

	except:
		if list_of_lists_flag:
			valList = []
			for listIdx in range(lenHighest):
				valList.append(default[0])
		else:
			valList = default

	return valList


def parse_value(expr):
	"""
	Parse read text value into dict value
	"""

	try:
		return eval(expr)
	except:
		return eval(re.sub("\s+", ",", expr))
	else:
		return expr


def parse_line(line):
	"""
	Parse read text line into dict key and value
	"""

	eq = line.find('=')
	if eq == -1: raise Exception()
	key = line[:eq].strip()
	value = line[eq+1:-1].strip()
	return key, parse_value(value)


def read_input_file(input_file):
	"""
	Read input file
	"""

	# TODO: better exception handling besides just a pass

	read_dict = {}
	with open(input_file) as f:
		contents = f.readlines()

	for line in contents: 
		try:
			key, val = parse_line(line)
			read_dict[key] = val
			# convert lists to NumPy arrays
			if (type(val) == list): 
				read_dict[key] = np.asarray(val)
		except:
			pass 

	return read_dict


def parse_bc(bc_name, in_dict):
	"""
	Parse boundary condition parameters from the input parameter dictionary
	"""

	# TODO: can definitely be made more general

	if ("press_"+bc_name in in_dict): 
		press = in_dict["press_"+bc_name]
	else:
		press = None 
	if ("vel_"+bc_name in in_dict): 
		vel = in_dict["vel_"+bc_name]
	else:
		vel = None 
	if ("temp_"+bc_name in in_dict):
		temp = in_dict["temp_"+bc_name]
	else:
		temp = None 
	if ("mass_frac_"+bc_name in in_dict):
		mass_frac = in_dict["mass_frac_"+bc_name]
	else:
		mass_frac = None
	if ("rho_"+bc_name in in_dict):
		rho = in_dict["rho_"+bc_name]
	else:
		rho = None
	if ("pert_type_"+bc_name in in_dict):
		pert_type = in_dict["pert_type_"+bc_name]
	else:
		pert_type = None
	if ("pert_perc_"+bc_name in in_dict):
		pert_perc = in_dict["pert_perc_"+bc_name]
	else:
		pert_perc = None
	if ("pert_freq_"+bc_name in in_dict):
		pert_freq = in_dict["pert_freq_"+bc_name]
	else:
		pert_freq = None
	
	return press, vel, temp, mass_frac, rho, pert_type, pert_perc, pert_freq


def get_initial_conditions(sol_domain, solver):
	"""
	Extract initial condition profile from two-zone initParamsFile, init_file .npy file, or restart file
	"""

	# TODO: add an option to interpolate a solution onto the given mesh, if different

	# intialize from restart file
	if solver.init_from_restart:
		solver.sol_time, sol_prim_init, solver.restart_iter = read_restart_file()

	# otherwise init from scratch IC or custom IC file 
	else:
		if (solver.init_file == None):
			sol_prim_init = gen_piecewise_uniform_ic(sol_domain, solver)
		else:
			# TODO: change this to .npz format with physical time included
			sol_prim_init = np.load(solver.init_file)
			assert (sol_prim_init.shape[0] == sol_domain.gas_model.num_eqs), ("Incorrect init_file num_eqs: "+str(sol_prim_init.shape[0]))

		# attempt to get solver.sol_time, if given
		solver.sol_time = catch_input(solver.param_dict, "sol_time_init", 0.0)

	return sol_prim_init


def gen_piecewise_uniform_ic(sol_domain, solver):
	"""
	Generate "left" and "right" states
	"""

	# TODO: generalize to >2 uniform regions

	if os.path.isfile(solver.ic_params_file):
		ic_dict 	= read_input_file(solver.ic_params_file)
	else:
		raise ValueError("Could not find initial conditions file at "+solver.ic_params_file)

	split_idx 	= np.absolute(solver.mesh.x_cell - ic_dict["xSplit"]).argmin()+1
	sol_prim 	= np.zeros((sol_domain.gas_model.num_eqs, solver.mesh.num_cells), dtype=const.REAL_TYPE)

	# TODO: error (or warning?) if xSplit is outside domain / doesn't split domain at all

	gas = sol_domain.gas_model

	# left state
	sol_prim[0,:split_idx] 	= ic_dict["press_left"]
	sol_prim[1,:split_idx] 	= ic_dict["vel_left"]
	sol_prim[2,:split_idx] 	= ic_dict["temp_left"]
	mass_frac_left          = ic_dict["mass_frac_left"]
	assert(np.sum(mass_frac_left) == 1.0), "mass_frac_left must sum to 1.0"
	assert(len(mass_frac_left) == gas.num_species_full), "mass_frac_left must have "+str(gas.num_species_full)+" entries"
	sol_prim[3:,:split_idx] 	= ic_dict["mass_frac_left"][gas.mass_frac_slice, None]

	# right state
	sol_prim[0,split_idx:] 	= ic_dict["press_right"]
	sol_prim[1,split_idx:] 	= ic_dict["vel_right"]
	sol_prim[2,split_idx:] 	= ic_dict["temp_right"]
	mass_frac_right         = ic_dict["mass_frac_right"]
	assert(np.sum(mass_frac_right) == 1.0), "mass_frac_right must sum to 1.0"
	assert(len(mass_frac_right) == gas.num_species_full), "mass_frac_right must have "+str(gas.num_species_full)+" entries"
	sol_prim[3:,split_idx:] 	= mass_frac_right[gas.mass_frac_slice, None]
	
	return sol_prim


def read_restart_file():
	"""
	Read solution state from restart file 
	"""

	# TODO: if higher-order multistep scheme, load previous time steps to preserve time accuracy

	# read text file for restart file iteration number
	with open(os.path.join(const.restart_output_dir, "restart_iter.dat"), "r") as f:
		restart_iter = int(f.read())

	# read solution
	restartFile = os.path.join(const.restart_output_dir, "restartFile_"+str(restart_iter)+".npz")
	restart_in = np.load(restartFile)

	sol_time = restart_in["sol_time"].item() 	# convert array() to scalar
	sol_prim = restart_in["sol_prim"]

	restart_iter += 1 # so this restart file doesn't get overwritten on next restart write

	return sol_time, sol_prim, restart_iter
import os
import re

import numpy as np

from perform.constants import REAL_TYPE


def catch_input(in_dict, in_key, default_val):
    """
    Assign default values if user does not provide a certain input
    """

    # TODO: correct error handling if default type is not recognized
    # TODO: check against lowercase'd strings
    # 		so that inputs are not case sensitive.
    # 		Do this for True/False too
    # TODO: instead of trusting user for None,
    # 		could also use NaN/Inf to indicate int/float defaults
    # 		without passing a numerical default
    # 		Or could just pass the actual default type lol, that'd be easier

    try:
        # If None passed as default, trust user
        if default_val is None:
            outVal = in_dict[in_key]
        else:
            default_type = type(default_val)
            outVal = default_type(in_dict[in_key])
    except KeyError:
        outVal = default_val

    return outVal


def catch_list(in_dict, in_key, default, len_highest=1):
    """
    Input processor for reading lists or lists of lists
    Default defines length of lists at lowest level
    """

    # TODO: needs to throw an error if input list of lists
    # 	is longer than len_highest
    # TODO: could make a recursive function probably,
    # 	just hard to define appropriate list lengths at each level

    list_of_lists_flag = type(default[0]) == list

    try:
        inList = in_dict[in_key]

        if len(inList) == 0:
            raise ValueError

        # List of lists
        if list_of_lists_flag:
            val_list = []
            for list_idx in range(len_highest):
                # If default value is None, trust user
                if default[0][0] is None:
                    val_list.append(inList[list_idx])
                else:
                    type_default = type(default[0][0])
                    cast_in_list = [type_default(inVal) for inVal in inList[list_idx]]
                    val_list.append(cast_in_list)

        # Normal list
        else:
            # If default value is None, trust user
            if default[0] is None:
                val_list = inList
            else:
                type_default = type(default[0])
                val_list = [type_default(inVal) for inVal in inList]

    except:
        if list_of_lists_flag:
            val_list = []
            for list_idx in range(len_highest):
                val_list.append(default[0])
        else:
            val_list = default

    return val_list


def parse_value(expr):
    """
    Parse read text value into dict value
    """

    try:
        return eval(expr)
    except:
        return eval(re.sub(r"\s+", ",", expr))
    else:
        return expr


def parse_line(line):
    """
    Parse read text line into dict key and value
    """

    eq = line.find("=")
    if eq == -1:
        raise Exception()
    key = line[:eq].strip()
    value = line[(eq + 1) : -1].strip()
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
            if isinstance(val, list):
                read_dict[key] = np.asarray(val)
        except:
            pass

    return read_dict


def parse_bc(bc_name, in_dict):
    """
    Parse boundary condition parameters from the input parameter dictionary
    """

    # TODO: can definitely be made more general

    if ("press_" + bc_name) in in_dict:
        press = in_dict["press_" + bc_name]
    else:
        press = None
    if ("vel_" + bc_name) in in_dict:
        vel = in_dict["vel_" + bc_name]
    else:
        vel = None
    if ("temp_" + bc_name) in in_dict:
        temp = in_dict["temp_" + bc_name]
    else:
        temp = None
    if ("mass_fracs_" + bc_name) in in_dict:
        mass_fracs = in_dict["mass_fracs_" + bc_name]
    else:
        mass_fracs = None
    if ("rho_" + bc_name) in in_dict:
        rho = in_dict["rho_" + bc_name]
    else:
        rho = None
    if ("pert_type_" + bc_name) in in_dict:
        pert_type = in_dict["pert_type_" + bc_name]
    else:
        pert_type = None
    if ("pert_perc_" + bc_name) in in_dict:
        pert_perc = in_dict["pert_perc_" + bc_name]
    else:
        pert_perc = None
    if ("pert_freq_" + bc_name) in in_dict:
        pert_freq = in_dict["pert_freq_" + bc_name]
    else:
        pert_freq = None

    return press, vel, temp, mass_fracs, rho, pert_type, pert_perc, pert_freq


def get_initial_conditions(sol_domain, solver):
    """
    Extract initial condition profile from
    two-zone initParamsFile, init_file .npy file, or restart file
    """

    # TODO: add option to interpolate solution onto given mesh, if different

    # Initialize from restart file
    if solver.init_from_restart:
        (solver.sol_time, sol_prim_init, solver.restart_iter) = read_restart_file(solver)

    # Otherwise init from scratch IC or custom IC file
    else:
        if solver.init_file is None:
            sol_prim_init = gen_piecewise_uniform_ic(sol_domain, solver)
        else:
            sol_prim_init = np.load(solver.init_file)
            assert sol_prim_init.shape[0] == sol_domain.gas_model.num_eqs, "Incorrect init_file num_eqs: " + str(
                sol_prim_init.shape[0]
            )

        # Attempt to get solver.sol_time, if given
        solver.sol_time = catch_input(solver.param_dict, "sol_time_init", 0.0)

    return sol_prim_init


def gen_piecewise_uniform_ic(sol_domain, solver):
    """
    Generate "left" and "right" states
    """

    # TODO: generalize to >2 uniform regions

    if os.path.isfile(solver.ic_params_file):
        ic_dict = read_input_file(solver.ic_params_file)
    else:
        raise ValueError("Could not find initial conditions file at " + solver.ic_params_file)

    split_idx = np.absolute(sol_domain.mesh.x_cell - ic_dict["x_split"]).argmin() + 1
    sol_prim = np.zeros((sol_domain.gas_model.num_eqs, sol_domain.mesh.num_cells), dtype=REAL_TYPE)

    # TODO: error (warning?) if x_split outside domain / doesn't split domain

    gas = sol_domain.gas_model

    # Left state
    sol_prim[0, :split_idx] = ic_dict["press_left"]
    sol_prim[1, :split_idx] = ic_dict["vel_left"]
    sol_prim[2, :split_idx] = ic_dict["temp_left"]
    mass_fracs_left = ic_dict["mass_fracs_left"]
    assert np.sum(mass_fracs_left) == 1.0, "mass_fracs_left must sum to 1.0"
    assert len(mass_fracs_left) == gas.num_species_full, (
        "mass_fracs_left must have " + str(gas.num_species_full) + " entries"
    )
    sol_prim[3:, :split_idx] = ic_dict["mass_fracs_left"][gas.mass_frac_slice, None]

    # Right state
    sol_prim[0, split_idx:] = ic_dict["press_right"]
    sol_prim[1, split_idx:] = ic_dict["vel_right"]
    sol_prim[2, split_idx:] = ic_dict["temp_right"]
    mass_fracs_right = ic_dict["mass_fracs_right"]
    assert np.sum(mass_fracs_right) == 1.0, "mass_fracs_right must sum to 1.0"
    assert len(mass_fracs_right) == gas.num_species_full, (
        "mass_fracs_right must have " + str(gas.num_species_full) + " entries"
    )
    sol_prim[3:, split_idx:] = mass_fracs_right[gas.mass_frac_slice, None]

    return sol_prim


def read_restart_file(solver):
    """
    Read solution state from restart file
    """

    # TODO: if higher-order multistep scheme,
    # 	load previous time steps to preserve time accuracy

    # Read text file for restart file iteration number
    iter_file = os.path.join(solver.restart_output_dir, "restart_iter.dat")
    with open(iter_file, "r") as f:
        restart_iter = int(f.read())

    # Read solution
    restart_file = os.path.join(solver.restart_output_dir, "restartFile_" + str(restart_iter) + ".npz")
    restart_in = np.load(restart_file)

    sol_time = restart_in["sol_time"].item()  # convert array() to scalar
    sol_prim = restart_in["sol_prim"]

    restart_iter += 1  # so this restart file doesn't get overwritten

    return sol_time, sol_prim, restart_iter

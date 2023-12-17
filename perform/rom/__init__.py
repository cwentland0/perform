from packaging import version
from time import sleep
import numpy as np
from numpy.linalg import svd
import scipy.linalg as LA
import sys

# ROM methods
from perform.rom.rom_method.projection_method.galerkin_projection import GalerkinProjection
from perform.rom.rom_method.projection_method.lspg_projection import LSPGProjection
from perform.rom.rom_method.projection_method.mplsvt_projection import MPLSVTProjection

# variable mappings
from perform.rom.rom_variable_mapping.prim_variable_mapping import PrimVariableMapping
from perform.rom.rom_variable_mapping.cons_variable_mapping import ConsVariableMapping

# time steppers
from perform.rom.rom_time_stepper.numerical_stepper import NumericalStepper

# space mappings
from perform.rom.rom_space_mapping.linear_space_mapping import LinearSpaceMapping

# Check whether ML libraries are accessible
# Tensorflow-Keras
TFKERAS_IMPORT_SUCCESS = True
try:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # don't print all the TensorFlow warnings
    import tensorflow as tf

    MLVER = tf.__version__
    from perform.rom.ml_library.tfkeras_library import TFKerasLibrary

except ImportError:
    TFKERAS_IMPORT_SUCCESS = False

# PyTorch
TORCH_IMPORT_SUCCESS = True
try:
    import torch

    MLVER = torch.__version__
    from perform.rom.ml_library.pytorch_library import PyTorchLibrary

except ImportError:
    TORCH_IMPORT_SUCCESS = False


def get_rom_method(rom_method, sol_domain, solver, rom_domain):

    if rom_method == "galerkin":
        return GalerkinProjection(sol_domain, rom_domain, solver)
    elif rom_method == "lspg":
        return LSPGProjection(sol_domain, rom_domain, solver)
    elif rom_method == "mplsvt":
        return MPLSVTProjection(sol_domain, rom_domain, solver)
    else:
        raise ValueError("Invalid ROM rom_method: " + str(rom_method))


def get_variable_mapping(var_mapping, sol_domain, rom_domain):

    if var_mapping == "primitive":
        return PrimVariableMapping(sol_domain, rom_domain)
    elif var_mapping == "conservative":
        return ConsVariableMapping(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM var_mapping: " + str(var_mapping))


def get_time_stepper(time_stepper, sol_domain, solver, rom_domain):

    if time_stepper == "numerical":
        return NumericalStepper(sol_domain, rom_domain, solver)
    else:
        raise ValueError("Something went wrong, rom_method set invalid time_stepper: " + str(time_stepper))


def get_space_mapping(space_mapping, sol_domain, rom_domain):

    if space_mapping == "linear":
        return LinearSpaceMapping(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM space_mapping: " + str(space_mapping))


def get_ml_library(ml_library, rom_domain):

    if ml_library == "tfkeras":
        assert TFKERAS_IMPORT_SUCCESS, "Tensorflow failed to import, please check that it is installed"
        if version.parse(tf.__version__) < version.parse("2.4.1"):
            print("WARNING: You are using TensorFlow version < 2.4.1, proper ROM behavior not guaranteed")
            sleep(1.0)
        return TFKerasLibrary(rom_domain)

    elif ml_library == "pytorch":
        assert TORCH_IMPORT_SUCCESS, "PyTorch failed to import, please check that it is installed."
        return PyTorchLibrary(rom_domain)

    else:
        raise ValueError("Invalid mllib_name: " + str(ml_library))

def gen_rom_basis(data_dir, dt, iter_start, iter_end, iter_skip, cent_type, norm_type, var_idxs, max_modes, rom_method):

    # construct data file
    if rom_method == "mplsvt":
        data_file = "unsteady_field_results/sol_prim_FOM_dt_" + str(dt) + ".npy"
        data_file_cons = "unsteady_field_results/sol_cons_FOM_dt_" + str(dt) + ".npy"
    else:
        data_file = "unsteady_field_results/sol_cons_FOM_dt_" + str(dt) + ".npy"
    
    # load data, subsample
    in_file = os.path.join(data_dir, data_file)
    snap_arr = np.load(in_file)

    snap_arr = snap_arr[:, :, iter_start : iter_end + 1 : iter_skip]
    if rom_method == "mplsvt":
        in_file_cons = os.path.join(data_dir, data_file_cons)
        snap_arr_cons = np.load(in_file_cons)
        snap_arr_cons = snap_arr_cons[:, :, iter_start : iter_end + 1 : iter_skip]
    snap_arr_noskip = snap_arr[:, :, iter_start : iter_end + 1]
    _, num_cells, num_snaps = snap_arr.shape

    spatial_modes = []
    cent_file = []
    norm_sub_file = []
    norm_fac_file = []
    cent_file_cons = []
    norm_sub_file_cons = []
    norm_fac_file_cons = []

    singval_states = []
    
    # loop through groups
    for group_idx, var_idx_list in enumerate(var_idxs):

        print("ROM basis processing variable group " + str(group_idx + 1))

        # break data array into different variable groups
        group_arr = snap_arr[var_idx_list, :, :]
        if rom_method == "mplsvt":
            group_arr_cons = snap_arr_cons[var_idx_list, :, :]
        group_arr_noskip = snap_arr_noskip[var_idx_list, :, :]
        num_vars = group_arr.shape[0]

        # center and normalize data
        group_arr, cent_prof = center_data(group_arr, cent_type)
        group_arr, norm_sub_prof, norm_fac_prof = normalize_data(group_arr, norm_type)
        if rom_method == "mplsvt":
            group_arr_cons, cent_prof_cons = center_data(group_arr_cons, cent_type)
            _, norm_sub_prof_cons, norm_fac_prof_cons = normalize_data(group_arr_cons, norm_type)
        
        group_arr_noskip, _ = center_data(group_arr_noskip, cent_type)
        group_arr_noskip, _, _ = normalize_data(group_arr_noskip, norm_type)

        min_dim = min(num_cells * num_vars, num_snaps)
        modes_out = min(min_dim, max_modes[group_idx])

        # compute SVD
        group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
        U, singval, VT = svd(group_arr)
        U = np.reshape(U, (num_vars, num_cells, U.shape[-1]), order="C")
        
        group_arr_noskip = np.reshape(group_arr_noskip, (-1, group_arr_noskip.shape[-1]), order="C")

        # truncate modes
        basis = U[:, :, :modes_out]

        spatial_modes.append(basis)
        cent_file.append(cent_prof)
        norm_sub_file.append(norm_sub_prof)
        norm_fac_file.append(norm_fac_prof)
        if rom_method == "mplsvt":
            cent_file_cons.append(cent_prof_cons)
            norm_sub_file_cons.append(norm_sub_prof_cons)
            norm_fac_file_cons.append(norm_fac_prof_cons)
        
        # compute singular value of each degree of freedom
        if len(var_idxs) == 1 and group_idx == 0:
            
            for idx in range(num_vars):
                snap_states = group_arr[idx*num_cells:(idx + 1)*num_cells,:]
                _, Sv_states, _ = np.linalg.svd(snap_states)
                singval_states.append(Sv_states)
            
            singval_states = np.asarray(singval_states)
        
    print("POD basis generated!")
    
    return spatial_modes, cent_file, norm_sub_file, norm_fac_file, singval, singval_states, group_arr_noskip, cent_file_cons, norm_sub_file_cons, norm_fac_file_cons

def gen_deim_sampling(var_idxs, basis, deim_dim):
    
    assert len(var_idxs) == 1, "Non-vector rom not implemented yet for DEIM"

    # find number of nodes
    n_nodes = basis.shape[1]

    # reshape the basis matrix
    group_arr = basis[var_idxs[0], :, :]
    group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
    
    # perform qr with pivoting
    _, _, sampling = LA.qr(group_arr.T, pivoting=True)
    sampling_trunc = sampling[:deim_dim]

    # apply modulo
    sampling_id = np.remainder(sampling_trunc, n_nodes)
    sampling_id = np.unique(sampling_id)
    
    ctr = 0
    while sampling_id.shape[0] < deim_dim:
        # get the next sampling index
        sampling_id = np.append(sampling_id, np.remainder(sampling[deim_dim + ctr], n_nodes))
        
        # ensure all entries are unique
        sampling_id = np.unique(sampling_id)
        ctr = ctr + 1
    
    # sort indices
    sampling_id = np.sort(sampling_id)  

    print("DEIM sampling points generated!")
    return sampling_id

# center training data
def center_data(data_arr, cent_type):

    # center around the initial condition
    if cent_type == "init_cond":
        cent_prof = data_arr[:, :, [0]]

    # center around the sample mean
    elif cent_type == "mean":
        cent_prof = np.mean(data_arr, axis=2, keepdims=True)

    else:
        raise ValueError("Invalid cent_type input: " + str(cent_type))

    data_arr -= cent_prof

    return data_arr, np.squeeze(cent_prof, axis=-1)


# normalize training data
def normalize_data(data_arr, norm_type):

    ones_prof = np.ones((data_arr.shape[0], data_arr.shape[1], 1), dtype=np.float64)
    zero_prof = np.zeros((data_arr.shape[0], data_arr.shape[1], 1), dtype=np.float64)

    # normalize by  (X - min(X)) / (max(X) - min(X))
    if norm_type == "minmax":
        min_vals = np.amin(data_arr, axis=(1, 2), keepdims=True)
        max_vals = np.amax(data_arr, axis=(1, 2), keepdims=True)
        norm_sub_prof = min_vals * ones_prof
        norm_fac_prof = (max_vals - min_vals) * ones_prof

    # normalize by L2 norm sqaured of each variable
    elif norm_type == "l2":
        data_arr_sq = np.square(data_arr)
        norm_fac_prof = np.sum(np.sum(data_arr_sq, axis=1, keepdims=True), axis=2, keepdims=True)
        norm_fac_prof /= data_arr.shape[1] * data_arr.shape[2]
        norm_fac_prof = norm_fac_prof * ones_prof
        norm_sub_prof = zero_prof

    else:
        raise ValueError("Invalid norm_type input: " + str(norm_type))

    data_arr = (data_arr - norm_sub_prof) / norm_fac_prof

    return data_arr, np.squeeze(norm_sub_prof, axis=-1), np.squeeze(norm_fac_prof, axis=-1)

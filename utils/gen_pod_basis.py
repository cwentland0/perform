import os

import numpy as np
from numpy.linalg import svd


##### BEGIN USER INPUT #####

data_dir = "~/path/to/data/dir"
data_file = "sol_prim_FOM.npy"

iter_start = 0  # zero-indexed starting index for snapshot array
iter_end = 4000  # zero-indexed ending index for snapshot array
iter_skip = 1

# centering method, accepts "init_cond" and "mean"
cent_type = "init_cond"

# normalization method, accepts "minmax" and "l2"
norm_type = "minmax"

# zero-indexed list of lists for group variables
var_idxs = [[0], [1], [2], [3]]

max_modes = 25

out_dir = "~/path/to/output/dir"

##### END USER INPUT #####

out_dir = os.path.expanduser(out_dir)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


def main():

    # load data, subsample
    in_file = os.path.join(data_dir, data_file)
    snap_arr = np.load(in_file)
    snap_arr = snap_arr[:, :, iter_start : iter_end + 1 : iter_skip]
    _, num_cells, num_snaps = snap_arr.shape

    # loop through groups
    for group_idx, var_idx_list in enumerate(var_idxs):

        print("Processing variable group " + str(group_idx + 1))

        # break data array into different variable groups
        group_arr = snap_arr[var_idx_list, :, :]
        num_vars = group_arr.shape[0]

        # center and normalize data
        group_arr, cent_prof = center_data(group_arr)
        group_arr, norm_sub_prof, norm_fac_prof = normalize_data(group_arr)

        min_dim = min(num_cells * num_vars, num_snaps)
        modes_out = min(min_dim, max_modes)

        # compute SVD
        group_arr = np.reshape(group_arr, (-1, group_arr.shape[-1]), order="C")
        U, s, VT = svd(group_arr)
        U = np.reshape(U, (num_vars, num_cells, U.shape[-1]), order="C")

        # truncate modes
        basis = U[:, :, :modes_out]

        # suffix for output files
        suffix = ""
        for var_idx in var_idx_list:
            suffix += "_" + str(var_idx)
        suffix += ".npy"

        # save data to disk
        cent_file = os.path.join(out_dir, "cent_prof")
        norm_sub_file = os.path.join(out_dir, "norm_sub_prof")
        norm_fac_file = os.path.join(out_dir, "norm_fac_prof")
        spatial_mode_file = os.path.join(out_dir, "spatial_modes")
        sing_vals_file = os.path.join(out_dir, "singular_values")

        np.save(cent_file + suffix, cent_prof)
        np.save(norm_sub_file + suffix, norm_sub_prof)
        np.save(norm_fac_file + suffix, norm_fac_prof)
        np.save(spatial_mode_file + suffix, basis)
        np.save(sing_vals_file + suffix, s)

    print("POD basis generated!")


# center training data
def center_data(data_arr):

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
def normalize_data(data_arr):

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
        raise ValueError("Invalid norm_type input: " + str(cent_type))

    data_arr = (data_arr - norm_sub_prof) / norm_fac_prof

    return data_arr, np.squeeze(norm_sub_prof, axis=-1), np.squeeze(norm_fac_prof, axis=-1)


if __name__ == "__main__":
    main()

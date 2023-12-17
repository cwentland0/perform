"""Miscellaneous functions that do not have a clear place in other modules."""

import os
import numpy as np
import scipy.linalg as LA


def mkdir_shallow(base_dir, new_dir_name):
    """Makes new directory, if it does not already exist.

    Just a helper function for handling path joining.

    Args:
        base_dir: Path to directory in which new directory is to be made.
        new_dir_name: Name of new directory to be created in base_dir.

    Returns:
        Path to new directory.
    """

    new_dir = os.path.join(base_dir, new_dir_name)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    return new_dir

def deim_helper(group_arr, deim_dim, n_nodes):
    
    # perform qr with pivoting
    _, _, qdeim_pivots = LA.qr(group_arr.T, pivoting=True)
    retained_pivots = qdeim_pivots[:deim_dim]

    # apply modulo
    sampling_id = np.remainder(retained_pivots, n_nodes)
    sampling_id = np.unique(sampling_id)
    
    ctr = 0
    while sampling_id.shape[0] < deim_dim:
        # get the next sampling index
        sampling_id = np.append(sampling_id, np.remainder(qdeim_pivots[deim_dim + ctr], n_nodes))
        
        # ensure all entries are unique
        sampling_id = np.unique(sampling_id)
        ctr = ctr + 1
    
    # sort indices
    sampling_id = np.sort(sampling_id)  

    return sampling_id

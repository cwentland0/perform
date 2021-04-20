"""Miscellaneous functions that do not have a clear place in other modules."""

import os
import struct


def write_to_file(fid, array, order="F"):
    """Write NumPy arrays to binary file using struct.

    I used to use this for comparisons against GEMS, but is not
    terribly useful otherwise. Kept here just in case.

    Args:
        fid: File opened with open(), in mode 'wb'.
        array: Array to be written to fid.
        order: "C" or "F", order in which array will be flattened for output. "C" is row-major, "F" is column-major.
    """

    if array.ndim > 1:
        array = array.flatten(order=order)
    dtype = array.dtype
    if dtype == "float64":
        typeStr = "d"
    elif dtype == "float32":
        typeStr = "f"
    elif dtype == "int32":
        typeStr = "i"
    elif dtype == "int16":
        typeStr = "h"
    else:
        raise ValueError("Did not recognize array type " + dtype)

    fid.write(struct.pack(typeStr * array.shape[0], *(array)))


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

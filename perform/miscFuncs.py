import os
import struct

import perform.constants as const


def write_to_file(fid, array, order='F'):
	"""
	Write NumPy arrays to binary file using struct

	Inputs
	------
	fid : file object
		File opened with open(), in mode 'wb'
	array : ndarray
		Array to be written to fid
	order (optional) : 'C' or 'F'
		Order in which array will be flattened for output
		'C' is row-major, 'F' is column-major

	Outputs
	-------
	None

	"""

	if (array.ndim > 1):
		array = array.flatten(order=order)
	dtype = array.dtype
	if (dtype == "float64"):
		typeStr = "d"
	elif (dtype == "float32"):
		typeStr = "f"
	elif (dtype == "int32"):
		typeStr = "i"
	elif (dtype == "int16"):
		typeStr = "h"
	else:
		raise ValueError("Did not recognize array type " + dtype)

	fid.write(struct.pack(typeStr * array.shape[0], *(array)))


def mkdir_shallow(base_dir, new_dir_name):

	new_dir = os.path.join(base_dir, new_dir_name)
	if not os.path.isdir(new_dir):
		os.mkdir(new_dir)
	return new_dir

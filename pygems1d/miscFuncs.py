import constants
import os
import struct

def writeToFile(fid, array, order='F'):
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
		raise ValueError ("Did not recognize array type "+dtype)

	fid.write(struct.pack(typeStr*array.shape[0], *(array)))


def mkdirInWorkdir(dirName):

	newDir = os.path.join(constants.workingDir, dirName)
	if not os.path.isdir(newDir): os.mkdir(newDir)
	return newDir
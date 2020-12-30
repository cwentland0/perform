import struct

# write NumPy arrays to binary file
# add new typeStr for different dtypes as needed
def writeToFile(fid, array, order='F'):
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

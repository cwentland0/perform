import numpy as np 
from numpy.linalg import svd
import pdb
import os

##### BEGIN USER INPUT #####

dataDir 	= "~/path/to/data/dir"
dataFile 	= "solCons_FOM.npy"
iterStart 	= 0 		# zero-indexed starting index for snapshot array
iterEnd 	= 700		# zero-indexed ending index for snapshot array
iterSkip 	= 1

centType 	= "initCond" 		# accepts "initCond" and "mean"
normType 	= "l2"				# accepts "minmax" and "l2"

# zero-indexed list of lists for group variables
varIdxs 	= [[0],[1],[2],[3]]	

maxModes 	= 700

writeRightEvecs = False

basisOutDir = "podData_prim_press_vel_temp_mf_samp1"

##### END USER INPUT #####

outDir = os.path.join(os.path.expanduser(dataDir), basisOutDir)
if not os.path.isdir(outDir): os.mkdir(outDir)

def main():

	# load data
	inFile = os.path.join(dataDir, dataFile)
	snapArr = np.load(inFile)
	snapArr = snapArr[:,:,iterStart:iterEnd+1:iterSkip] 	# subsample
	nVarsTot, nCells, nSnaps = snapArr.shape

	# loop through groups
	for groupIdx, varIdxList in enumerate(varIdxs):

		groupArr = snapArr[varIdxList,:,:]	# break data array into different variable groups
		nVars = groupArr.shape[0]

		# center and normalize data 
		groupArr, centProf = centerData(groupArr)
		groupArr, normSubProf, normFacProf = normalizeData(groupArr)		

		minDim = min(nCells*nVars, nSnaps)
		modesOut = min(minDim, maxModes)

		# compute SVD 
		groupArr = np.reshape(groupArr, (-1, groupArr.shape[-1]), order="C")
		U, s, VT = svd(groupArr)
		U = np.reshape(U, (nVars, nCells, U.shape[-1]), order="C")
		basis = U[:,:,:modesOut] # truncate modes

		# suffix for output files
		suffix = ""
		for varIdx in varIdxList:
			suffix += "_" + str(varIdx)
		suffix += ".npy"	

		# save data to disk
		centFile 		= os.path.join(outDir, "centProf")
		normSubFile 	= os.path.join(outDir, "normSubProf")
		normFacFile 	= os.path.join(outDir, "normFacProf")
		spatialModeFile = os.path.join(outDir, "spatialModes")
		singValsFile	= os.path.join(outDir, "singularValues")

		np.save(centFile+suffix, centProf)
		np.save(normSubFile+suffix, normSubProf)
		np.save(normFacFile+suffix, normFacProf)
		np.save(spatialModeFile+suffix, basis)
		np.save(singValsFile+suffix, s)

	print("POD basis generated!")

# center training data
def centerData(dataArr):

	# center around the initial condition
	if (centType == "initCond"):
		centProf = dataArr[:,:,[0]]

	# center around the sample mean
	elif (centType == "mean"):
		centProf = np.mean(dataArr, axis=2, keepdims=True)

	else:
		raise ValueError("Invalid centType input: "+str(centType))

	dataArr -= centProf

	return dataArr, np.squeeze(centProf, axis=-1)


# normalize training data
def normalizeData(dataArr):

	onesProf = np.ones((dataArr.shape[0],dataArr.shape[1],1), dtype = np.float64)
	zeroProf = np.zeros((dataArr.shape[0],dataArr.shape[1],1), dtype = np.float64)

	# normalize by  (X - min(X)) / (max(X) - min(X)) 
	if (normType == "minmax"):
		minVals = np.amin(dataArr, axis=(1,2), keepdims=True)
		maxVals = np.amax(dataArr, axis=(1,2), keepdims=True)
		normSubProf = minVals * onesProf
		normFacProf = (maxVals - minVals) * onesProf

	# normalize by L2 norm sqaured of each variable
	elif (normType == "l2"):
		dataArrSq = np.square(dataArr)
		normFacProf = np.sum(np.sum(dataArrSq, axis=1, keepdims=True), axis=2, keepdims=True) 
		normFacProf /= (dataArr.shape[1] * dataArr.shape[2])
		normFacProf = normFacProf * onesProf
		normSubProf = zeroProf

	else: 
		raise ValueError("Invalid normType input: "+str(centType))

	dataArr = (dataArr - normSubProf) / normFacProf 

	return dataArr, np.squeeze(normSubProf, axis = -1), np.squeeze(normFacProf, axis = -1)


if __name__ == "__main__":
	main()
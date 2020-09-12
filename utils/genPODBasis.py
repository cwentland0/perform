import numpy as np 
from numpy.linalg import svd
import matplotlib.pyplot as plt
import pdb
import os

# TODO: spatial modes should be lists too, to accommodate different latent dims for each basis

##### BEGIN USER INPUT #####

dataDir 	= "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/standingFlame/dataProc/40microsec"
dataFile 	= "solCons_FOM.npy"		
iterStart 	= 1 		# one-indexed starting index for snapshot array
iterEnd 	= 4001		# one-indexed ending index for snapshot array
iterSkip 	= 1

centType 	= "initCond" 		# accepts "initCond" and "mean"
normType 	= "l2"			# accepts "minmax"

varIdxs 	= [[0,1,2,3]]	# zero-indexed list of lists for group variables

maxModes 	= 200

writeRightEvecs = False

##### END USER INPUT #####

def main():

	# load data
	inFile = os.path.join(dataDir, dataFile)
	snapArr = np.load(inFile)
	snapArr = snapArr[:,:,iterStart-1:iterEnd:iterSkip] 	# subsample
	nCells, nVarsTot, nSnaps = snapArr.shape

	minDim = min(nCells*nVarsTot, nSnaps)
	assert(maxModes <= minDim)

	# loop through groups
	basisOut 	 	= np.zeros((nCells, nVarsTot, maxModes), dtype = np.float64)
	singVals 		= np.zeros((minDim,len(varIdxs)), dtype = np.float64)
	centProfOut 	= np.zeros((nCells, nVarsTot), dtype = np.float64)
	normSubOut 		= np.zeros((nCells, nVarsTot), dtype = np.float64)
	normFacOut 		= np.zeros((nCells, nVarsTot), dtype = np.float64)
	for groupIdx, varIdxList in enumerate(varIdxs):

		groupArr = snapArr[:,varIdxList,:]	# break data array into different variable groups
		nVars = groupArr.shape[1]

		# center and normalize data 
		groupArr, centProf = centerData(groupArr)
		groupArr, normSubProf, normFacProf = normalizeData(groupArr)		

		# compute SVD 
		groupArr = np.reshape(groupArr, (-1, groupArr.shape[-1]), order="F")
		U, s, VT = svd(groupArr)
		U = np.reshape(U, (nCells, nVars, U.shape[-1]), order="F")

		# store arrays
		centProfOut[:,varIdxList] = centProf.copy()
		normSubOut[:,varIdxList] = normSubProf.copy() 
		normFacOut[:,varIdxList] = normFacProf.copy()
		basisOut[:,varIdxList,:] = U[:,:,:maxModes] # truncate modes
		singVals[:,groupIdx] = s

	# suffix for output files
	suffix = ""
	for varIdxList in varIdxs:
		suffix += "_"
		for varIdx in varIdxList:
			suffix += str(varIdx)
	suffix += ".npy"	

	pdb.set_trace()

	# save data to disk
	outDir = os.path.join(dataDir, "podData")
	if not os.path.isdir(outDir): os.mkdir(outDir)

	centFile = os.path.join(outDir, "centProf")
	normSubFile = os.path.join(outDir, "normSubProf")
	normFacFile = os.path.join(outDir, "normFacProf")
	spatialModeFile = os.path.join(outDir, "spatialModes")
	singValsFile	= os.path.join(outDir, "singularValues")
	
	np.save(centFile+suffix, centProfOut)
	np.save(normSubFile+suffix, normSubOut)
	np.save(normFacFile+suffix, normFacOut)
	np.save(spatialModeFile+suffix, basisOut)
	np.save(singValsFile+suffix, singVals)

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

	return dataArr, np.squeeze(centProf)


# normalize training data
def normalizeData(dataArr):

	onesProf = np.ones((dataArr.shape[0],dataArr.shape[1],1), dtype = np.float64)
	zeroProf = np.zeros((dataArr.shape[0],dataArr.shape[1],1), dtype = np.float64)

	# normalize by  (X - min(X)) / (max(X) - min(X)) 
	if (normType == "minmax"):
		minVals = np.amin(dataArr, axis=(0,2), keepdims=True)
		maxVals = np.amax(dataArr, axis=(0,2), keepdims=True)
		normSubProf = minVals * onesProf
		normFacProf = (maxVals - minVals) * onesProf

	# normalize by L2 norm sqaured of each variable
	elif (normType == "l2"):
		dataArrSq = np.square(dataArr)
		normFacProf = np.sum(np.sum(dataArrSq, axis=0, keepdims=True), axis=2, keepdims=True) 
		normFacProf /= (dataArr.shape[0] * dataArr.shape[2])
		normFacProf = normFacProf * onesProf
		normSubProf = zeroProf

	else: 
		raise ValueError("Invalid normType input: "+str(centType))

	dataArr = (dataArr - normSubProf) / normFacProf 

	return dataArr, np.squeeze(normSubProf), np.squeeze(normFacProf)


if __name__ == "__main__":
	main()
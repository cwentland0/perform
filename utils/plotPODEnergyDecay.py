import numpy as np 
import matplotlib.pyplot as plt
import os
import pdb

##### BEGIN USER INPUT #####

dataDir = "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/standingFlame/dataProc/40microsec" 	# base dir where bases for particular dataset are stored
# singValsFile = "podData_cons_vector_test/singularValues_0123.npy"											# continuation of dataDir to singular values file
# singValsFile = "podData_cons_scalar_samp1/singularValues_0_1_2_3.npy"
singValsFile = "podData_cons_scalar_test/singularValues_0_123.npy"
lBound = 1
uBound = 200

##### END USER INPUT #####

# load and check bounds
inDir = os.path.join(dataDir, singValsFile)
singVals = np.load(inDir, allow_pickle=True)

jagged = False
# if jagged array, will come in as a list of arrays
if (singVals.ndim == 1):
	jagged = True
	numGroups = singVals.shape[0]
	numVals = np.zeros(numGroups, dtype=np.int32)
	for groupIdx in range(numGroups):
		numVals[groupIdx] = singVals[groupIdx].shape[0]
	assert(np.all(uBound <= numVals))

# if uniform array (only one group, or equally-sized groups), comes as a 2D array
else:
	numGroups, numVals = singVals.shape
	# pdb.set_trace()
	assert(uBound <= numVals)


assert(lBound >= 1)
assert(lBound < uBound)

# calculate energy decay for each group
fig = plt.figure()
ax = fig.add_subplot(111)
energy = np.zeros(uBound - lBound + 1, dtype = np.float64)
for groupIdx in range(numGroups):
	print("Group "+str(groupIdx+1))

	if jagged:
		singValsGroup = singVals[groupIdx]
	else:
		singValsGroup = singVals[groupIdx,:]

	sumSq = np.sum(np.square(singValsGroup))
	for sIdx in range(lBound-1, uBound):
		energy[sIdx] = 100. * (1. - np.sum(np.square(singValsGroup[:(sIdx+1)])) / sumSq)

	# pdb.set_trace()

	thresh99 = np.argwhere(energy < 1.0)[0][0] + 1
	thresh99p9 = np.argwhere(energy < 0.1)[0][0] + 1
	thresh99p99 = np.argwhere(energy < 0.01)[0][0] + 1

	print("99%: "+str(thresh99))
	print("99.9%: "+str(thresh99p9))
	print("99.99%: "+str(thresh99p99))

	ax.semilogy(range(lBound,uBound+1), energy)

	if (groupIdx == 0):
		minY = np.amin(energy)
	else:
		minY = min(np.amin(energy), minY)

if (minY < 1e-15): minY = 1e-16

ax.set_ylim([minY,100])

plt.show()
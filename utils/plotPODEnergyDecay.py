import numpy as np 
import matplotlib.pyplot as plt
import pdb

##### BEGIN USER INPUT #####

singValsFile = "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/standingFlame/dataProc/40microsec/podData_prim_samp10/singularValues_0123.npy"
lBound = 1
uBound = 200

##### END USER INPUT #####

# load and check bounds
singVals = np.load(singValsFile)
numVals, numGroups = singVals.shape
assert(lBound >= 1)
assert(uBound <= numVals)
assert(lBound < uBound)

# calculate energy decay for each group
fig = plt.figure()
ax = fig.add_subplot(111)
energy = np.zeros(uBound - lBound + 1, dtype = np.float64)
for groupIdx in range(numGroups):
	print("Group "+str(groupIdx+1))

	sumSq = np.sum(np.square(singVals[:,groupIdx]))
	for sIdx in range(lBound-1, uBound):
		energy[sIdx] = 100. * (1. - np.sum(np.square(singVals[:(sIdx+1),groupIdx])) / sumSq)

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
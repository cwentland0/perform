import numpy as np
from scipy.signal import periodogram 
import matplotlib.pyplot as plt
import pdb

##### BEGIN USER INPUT #####

probeData 	= "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/advectingFlame/ProbeResults/probe_pressure_velocity_FOM.npy"				
varIdx 		= 1 				# probe variable, 1-indexed
timeWindow	= [0.0001, 0.001]	# window to examine PSD for
dt  		= 1.0e-8
freqLims 	= [1, 1000000] 		# frequency window for PSD plot
outFile 	= "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/advectingFlame/ImageResults/psd_acousticTest.png"

##### END USER INPUT #####

# load data
probeData 	= np.load(probeData)

# extract time window
lBoundIdxs 	= np.argwhere(probeData[:,0] > timeWindow[0])
uBoundIdxs	= np.argwhere(probeData[:,0] <= timeWindow[1])
windowIdxs	= np.intersect1d(lBoundIdxs, uBoundIdxs) 

# extract signal, PSD
probeSignal = probeData[windowIdxs, varIdx]
# pdb.set_trace()
f, Pxx 		= periodogram(probeSignal, (1.0/dt))

# plot PSD
plt.semilogy(f[1:], Pxx[1:])
plt.xlim(freqLims)
plt.xlabel("Frequency (Hz)")
plt.ylabel('PSD (Pa^2/Hz)')

# save to disk, display
plt.savefig(outFile)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 02:17:53 2020

@author: ashis
"""

import numpy as np

file_path = './Dataset/'
scaled_file_path = './Scaled Dataset/'
sol_FOM = np.load(file_path+'solPrim_FOM.npy')
init_FOM = np.load(file_path+'ic_lf_256.npy')

#IC-centering 

sol_FOM = sol_FOM - np.repeat(init_FOM[:,:,0][:, :, np.newaxis], sol_FOM.shape[2], axis=2)

#scaling
max_FOM = np.amax(np.amax(sol_FOM,axis=0),axis=1)
min_FOM = np.amin(np.amin(sol_FOM,axis=0),axis=1)

for i in range(4):
    
    sol_FOM[:,i,:] = (sol_FOM[:,i,:] - min_FOM[i])/(max_FOM[i]-min_FOM[i])
    

scalers = np.zeros((4,2))
scalers[:,0] = max_FOM
scalers[:,1] = min_FOM

np.save(scaled_file_path+'solPrim_FOM_scaled.npy',sol_FOM)
np.save(scaled_file_path+'scalers.npy',scalers)


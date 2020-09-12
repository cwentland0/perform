# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:36:48 2020

@author: ashis
"""

from classDef_vis import viz_params
def calc_RAE(truth,pred):
    
    RAE = np.mean(np.abs(truth-pred))/np.max(np.abs(truth))
    
    return RAE


path_file = 'visualiseParams.txt'
params = viz_params(path_file)
params.visualize(params)




            
            
            

    
    

    

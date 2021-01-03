# useful constants used throughout the solver
import numpy as np

workingDir = None 	# working director, set at runtime

realType 	= np.float64 			# precision of real numbers
complexType = np.complex128 		# precision of complex numbers

RUniv 		= 8314.4621    # universal gas constant, J/(K*kmol)
enthRefTemp = 298.0 	# temperature at which reference enthalpy is measured
						# TODO: make this the default, but make it possible to accept different values in parameters
q0 			= 6.93e6
suthTemp 	= 110.4 	# Sutherland temperature

tinyNum 	= 1.0e-25 	# very small number
hugeNum 	= 1.0e25	# very large number

# "steady" state residual normalization defaults
resNormPrimDefault = [1.0e5, 10.0, 300.0, 1.0]

# implicit time integrator defaults
subiterMaxImpDefault 	= 50
l2ResTolDefault 		= 1.0e-12
dtauDefault 			= 1.0e-5
CFLDefault 				= 1.0
VNNDefault 				= 20.0

# output directory names
unsteadyOutputDirName   = "UnsteadyFieldResults"
probeOutputDirName 		= "ProbeResults"
imageOutputDirName 		= "ImageResults"
restartOutputDirName 	= "RestartFiles"

# working directory output directories, set at runtime
unsteadyOutputDir 	= None 
probeOutputDir 		= None 
imageOutputDir 		= None 
restartOutputDir 	= None

# input files
paramInputs 	= "solverParams.inp"
romInputs		= "romParams.inp"
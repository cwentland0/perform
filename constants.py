# useful constants used throughout the solver
import numpy as np

floatType = np.float64 	# precision of floating point numbers

RUniv 		= 8314.0    # universal gas constant, J/(K*kmol)
enthRefTemp = 298.0 	# temperature at which reference enthalpy is measured
						# TODO: make this the default, but make it possible to accept different values in parameters
q0 			= 6.93e6

tinyNum 	= 1.0e-25 	# small number for thresholding non-positive numbers to a small positive numbers

# time integration coefficients
rkCoeffs = np.array([0.25, 1.0/3.0, 0.5, 1.0], dtype = floatType)

# output directories
unsteadyOutputDir   = "UnsteadyFieldResults"
probeOutputDir 		= "ProbeResults"
imageOutputDir 		= "ImageResults"
restartOutputDir 	= "RestartFiles"

# input files
paramInputs 	= "solverParams.inp"
romInputs		= "romParams.inp"
# useful constants used throughout the solver
import numpy as np

floatType = np.float64 	# precision of floating point numbers

RUniv = 8314.0    # universal gas constant, J/(K*mol) * 1,000

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
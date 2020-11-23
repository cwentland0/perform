# useful constants used throughout the solver
import numpy as np

realType 	= np.float64 			# precision of real numbers
complexType = np.complex128 		# precision of complex numbers

RUniv 		= 8314.0    # universal gas constant, J/(K*kmol)
enthRefTemp = 298.0 	# temperature at which reference enthalpy is measured
						# TODO: make this the default, but make it possible to accept different values in parameters
q0 			= 6.93e6

tinyNum 	= 1.0e-25 	# small number for thresholding non-positive numbers to a small positive numbers

# "steady" state residual normalization defaults
steadyNormPrimDefault = [1.0e5, 10.0, 300.0, 1.0]

# time integration coefficients
rkCoeffs = np.array([0.25, 1.0/3.0, 0.5, 1.0], dtype = realType)

# BDF co-efficients
bdfCoeffs = np.array([1., 1.5, 11./6., 25./12.], dtype = realType)

# output directories
unsteadyOutputDir   = "UnsteadyFieldResults"
probeOutputDir 		= "ProbeResults"
imageOutputDir 		= "ImageResults"
restartOutputDir 	= "RestartFiles"

# input files
paramInputs 	= "solverParams.inp"
romInputs		= "romParams.inp"
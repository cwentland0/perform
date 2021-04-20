"""Useful constants used throughout the code"""

import numpy as np

# Precision of real and complex numbers
REAL_TYPE = np.float64
COMPLEX_TYPE = np.complex128

R_UNIV = 8314.4621  # universal gas constant, kJ/(K*mol)
SUTH_TEMP = 110.4  # Sutherland temperature

TINY_NUM = 1.0e-25  # very small number
HUGE_NUM = 1.0e25  # very large number

# time integrator defaults
SUBITER_MAX_IMP_DEFAULT = 50
L2_RES_TOL_DEFAULT = 1.0e-12
L2_STEADY_TOL_DEFAULT = 1.0e-12
RES_NORM_PRIM_DEFAULT = [1.0e5, 10.0, 300.0, 1.0]
DTAU_DEFAULT = 1.0e-5
CFL_DEFAULT = 1.0
VNN_DEFAULT = 20.0

FD_STEP_DEFAULT = 1.0e-6

# visualization constants
FIG_WIDTH_DEFAULT = 12
FIG_HEIGHT_DEFAULT = 6

# output directory names
UNSTEADY_OUTPUT_DIR_NAME = "unsteady_field_results"
PROBE_OUTPUT_DIR_NAME = "probe_results"
IMAGE_OUTPUT_DIR_NAME = "image_results"
RESTART_OUTPUT_DIR_NAME = "restart_files"

# input files
PARAM_INPUTS = "solver_params.inp"
ROM_INPUTS = "rom_params.inp"

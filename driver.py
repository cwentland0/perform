import argparse
import constants 
import os
from classDefs import parameters, gasProps, geometry 
from solver import solver
import pdb

# TODO: check for incorrect assignments that need a copy instead
# TODO: rename sol variable used to represent full domain state, our boundary sol name. Is confusing
# TODO: make code general for more than two species, array broadcasts are different for 2 vs 3+ species
#		idea: custom iterators for species-related slicing, or just squeeze any massfrac references
# read working directory input
parser = argparse.ArgumentParser(description = "Read working directory")
parser.add_argument('workdir', type = str, default = "./", help="runtime working directory")
args = parser.parse_args()
workdir = args.workdir

# make output directories
unsOutDir = os.path.join(workdir, constants.unsteadyOutputDir)
if not os.path.isdir(unsOutDir): os.mkdir(unsOutDir)
pointOutDir = os.path.join(workdir, constants.pointOutputDir)
if not os.path.isdir(pointOutDir): os.mkdir(pointOutDir)
imgOutDir = os.path.join(workdir, constants.imageOutputDir)
if not os.path.isdir(imgOutDir): os.mkdir(imgOutDir)

# read input files, setup problem
paramFile	= os.path.join(workdir, constants.paramFile)
paramIn 	= parameters(paramFile)
meshIn 		= geometry(paramIn.meshFile)
gasIn 		= gasProps(paramIn.gasFile)

# start run
solver(paramIn, meshIn, gasIn)

# pdb.set_trace()

print("Run finished!")
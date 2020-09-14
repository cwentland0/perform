import argparse
import constants 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # don't print all the TensorFlow warnings
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
workdir = os.path.expanduser(workdir) # expand tilde notation for home directory

# read input files, setup problem
paramFile	= os.path.join(workdir, constants.paramInputs)
paramIn 	= parameters(workdir, paramFile)
meshIn 		= geometry(paramIn.meshFile)
gasIn 		= gasProps(paramIn.gasFile)

# make output directories
if not os.path.isdir(paramIn.unsOutDir): 	os.mkdir(paramIn.unsOutDir)
if not os.path.isdir(paramIn.probeOutDir): 	os.mkdir(paramIn.probeOutDir)
if not os.path.isdir(paramIn.imgOutDir): 	os.mkdir(paramIn.imgOutDir)
if not os.path.isdir(paramIn.restOutDir): 	os.mkdir(paramIn.restOutDir)

# start run
solver(paramIn, meshIn, gasIn)

print("Run finished!")
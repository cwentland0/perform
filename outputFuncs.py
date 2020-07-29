import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
import os
from classDefs import parameters, geometry
from solution import solutionPhys
import constants
import time
import pdb

mpl.rc('font', family='serif',size='12')
mpl.rc('axes', labelsize='x-large')
mpl.rc('figure', facecolor='w')
mpl.rc('text', usetex=False)
mpl.rc('text.latex',preamble=r'\usepackage{amsmath}')

# TODO: could go overboard an make a visualization class
# TODO: plot/save multiple variables for probe and field/probe images
# TODO: add option to fix y-axis bounds?
# TODO: add visualization for arbitrary species
# TODO: move label selection to parameters
# TODO: better automatic formatting of images? Ultimately real plotting will be un utils, might now be worth the time

# store snapshots of field data
def storeFieldData(sol: solutionPhys, params:parameters, tStep):

	storeIdx = int(tStep / params.outInterval)

	if params.primOut: sol.primSnap[:,:,storeIdx] = sol.solPrim
	if params.consOut: sol.consSnap[:,:,storeIdx] = sol.solCons
	if params.RHSOut:  sol.RHSSnap[:,:,storeIdx]  = sol.RHS

def plotField(ax: plt.Axes, sol: solutionPhys, params: parameters, geom: geometry):

	ax.cla()

	if (params.visVar == "pressure"):
		field = sol.solPrim[:,0]
		axLabel = "Pressure (Pa)"
	elif (params.visVar == "velocity"):
		field = sol.solPrim[:,1]
		axLabel = "Velocity (m/s)"
	elif (params.visVar == "temperature"):
		field = sol.solPrim[:,2]
		axLabel = "Temperature (K)"
	elif (params.visVar == "species"):
		field = sol.solPrim[:,3]
		axLabel = "Species Mass Fraction"
	elif (params.visVar == "density"):
		field = sol.solCons[:,0]
		axLabel = "Density (kg/m^3)"
	elif (params.visVar == "momentum"):
		field = sol.solCons[:,1]
		axLabel = "Momentum (kg/s-m^2)"
	elif (params.visVar == "energy"):
		field = sol.solCons[:,2]
		axLabel = "Energy"
	else:
		raise ValueError("Invalid field visualization variable:"+str(params.visVar))

	ax.plot(geom.x_cell, field)
	ax.set_ylabel(axLabel)
	ax.set_xlabel("x (m)")
	# ax.set_ylim([290,310])
	# ax.set_xlim([0.0,0.0005])
	plt.subplots_adjust(left=0.2)
	plt.show(block=False)
	plt.pause(0.001)

# write field plot image to disk
def writeFieldImg(fig: plt.Figure, params: parameters, tStep, fieldImgDir):

	visIdx 	= int((tStep+1) / params.visInterval)
	figNum 	= params.imgString % visIdx
	figFile = os.path.join(fieldImgDir, "fig_"+figNum+".png")
	fig.savefig(figFile)

# update probeVals, as this happens every time iteration
def updateProbe(sol: solutionPhys, params: parameters, probeVals, probeIdx, tStep):

	if (params.visVar == "pressure"):
		probe = sol.solPrim[probeIdx,0]
	elif (params.visVar == "velocity"):
		probe = sol.solPrim[probeIdx,1]
	elif (params.visVar == "temperature"):
		probe = sol.solPrim[probeIdx,2]
	elif (params.visVar == "species"):
		probe = sol.solPrim[probeIdx,3]
	elif (params.visVar == "density"):
		probe = sol.solCons[probeIdx,0]
	elif (params.visVar == "momentum"):
		probe = sol.solCons[probeIdx,1]
	elif (params.visVar == "energy"):
		probe = sol.solCons[probeIdx,2]
	else:
		raise ValueError("Invalid field visualization variable:"+str(params.visVar))

	probeVals[tStep] = probe 

# plot probeVals at specied visInterval
def plotProbe(ax: plt.Axes, sol: solutionPhys, params: parameters, probeVals, tStep, tVals):

	ax.cla()

	if (params.visVar == "pressure"):
		axLabel = "Pressure (Pa)"
	elif (params.visVar == "velocity"):
		axLabel = "Velocity (m/s)"
	elif (params.visVar == "temperature"):
		axLabel = "Temperature (K)"
	elif (params.visVar == "species"):
		axLabel = "Species Mass Fraction"
	elif (params.visVar == "density"):
		axLabel = "Density (kg/m^3)"
	elif (params.visVar == "momentum"):
		axLabel = "Momentum (kg/s-m^2)"
	elif (params.visVar == "energy"):
		axLabel = "Energy"
	else:
		raise ValueError("Invalid field visualization variable:"+str(params.visVar))

	ax.plot(tVals[:tStep+1], probeVals[:tStep+1])
	ax.set_ylabel(axLabel)
	ax.set_xlabel("t (s)")
	# ax.set_ylim([975000,1025000])
	plt.subplots_adjust(left=0.2)
	plt.show(block=False)
	plt.pause(0.001)

# write snapshot matrices and point monitors to disk
def writeData(sol: solutionPhys, params: parameters, probeVals, tVals):

	# save snapshot matrices to disk
	if params.primOut:
		solPrimFile = os.path.join(params.unsOutDir, "solPrim_"+params.simType+".npy")
		np.save(solPrimFile, sol.primSnap)
	if params.consOut:
		solConsFile = os.path.join(params.unsOutDir, "solCons_"+params.simType+".npy")
		np.save(solConsFile, sol.consSnap) 
	if params.RHSOut:
		solRHSFile = os.path.join(params.unsOutDir, "solRHS_"+params.simType+".npy")
		np.save(solRHSFile, sol.RHSSnap) 

	# save point monitors to disk
	probeFile = os.path.join(params.probeOutDir, "probe_"+params.simType+".npy")
	probeSave = np.concatenate((tVals.reshape(-1,1), probeVals.reshape(-1,1)), axis=1)
	np.save(probeFile, probeSave)

# write restart files containing primitive and conservative fields, plus physical time 
def writeRestartFile(sol: solutionPhys, params: parameters, tStep):

	# write restart file to zipped file
	restartFile = os.path.join(params.restOutDir, "restartFile_"+str(params.restartIter)+".npz")
	np.savez(restartFile, solTime = params.solTime, solPrim = sol.solPrim, solCons = sol.solCons)

	# write iteration number files
	restartIterFile = os.path.join(params.restOutDir, "restartIter.dat")
	with open(restartIterFile, "w") as f:
		f.write(str(params.restartIter)+"\n")

	restartPhysIterFile = os.path.join(params.restOutDir, "restartIter_"+str(params.restartIter)+".dat")
	with open(restartPhysIterFile, "w") as f:
		f.write(str(tStep+1)+"\n")

	# iterate file count
	if (params.restartIter < params.numRestarts):
		params.restartIter += 1
	else:
		params.restartIter = 1
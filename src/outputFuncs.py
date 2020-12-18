import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
import os
from classDefs import parameters, geometry
from solution import solutionPhys, boundaries
import constants
import time
import pdb

mpl.rc('font', family='serif',size='10')
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
def storeFieldDataUnsteady(sol: solutionPhys, params:parameters, tStep):

	storeIdx = int(tStep / params.outInterval) + 1

	if params.primOut: sol.primSnap[:,:,storeIdx] = sol.solPrim
	if params.consOut: sol.consSnap[:,:,storeIdx] = sol.solCons
	if params.RHSOut:  sol.RHSSnap[:,:,storeIdx]  = sol.RHS

# get figure and axes handles for visualization
# TODO: some way to generalize this for any number of visualization plots?
# 	not even 9 variables to visualize right now
def setupPlotAxes(params: parameters):

	if (params.numVis == 1):
		params.visNRows = 1 
		params.visNCols = 1
	elif (params.numVis == 2):
		params.visNRows = 2
		params.visNCols = 1
	elif (params.numVis <= 4):
		params.visNRows = 2
		params.visNCols = 2
	elif (params.numVis <= 6):
		params.visNRows = 3
		params.visNCols = 2
	elif (params.numVis <= 9):
		params.visNRows = 3
		params.visNCols = 3

	# axis labels
	axLabels = []
	for axIdx in range(params.numVis):
		varStr = params.visVar[axIdx]
		if (varStr == "pressure"):
			axLabels.append("Pressure (Pa)")
		elif (varStr == "velocity"):
			axLabels.append("Velocity (m/s)")
		elif (varStr == "temperature"):
			axLabels.append("Temperature (K)")
		elif (varStr == "species"):
			axLabels.append("Species Mass Fraction")
		elif (varStr == "source"):
			axLabels.append("Reaction Source Term")
		elif (varStr == "density"):
			axLabels.append("Density (kg/m^3)")
		elif (varStr == "momentum"):
			axLabels.append("Momentum (kg/s-m^2)")
		elif (varStr == "energy"):
			axLabels.append("Total Energy")
		else:
			raise ValueError("Invalid field visualization variable:"+str(params.visVar))

	fig, ax = plt.subplots(nrows=params.visNRows, ncols=params.visNCols, figsize=(12,6))

	return fig, ax, axLabels

# plot field line plots
def plotField(fig: plt.Figure, ax: plt.Axes, axLabels, sol: solutionPhys, params: parameters, geom: geometry):

	if (type(ax) != np.ndarray): 
		axList = [ax]
	else:
		axList = ax 

	for colIdx, col in enumerate(axList):
		if (type(col) != np.ndarray):
			colList = [col]
		else:
			colList = col
		for rowIdx, axVar in enumerate(colList):

			axVar.cla()
			linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (params.visNRows, params.visNCols))[0]
			if ((linIdx+1) > params.numVis): 
				axVar.axis("off")
				break

			varStr = params.visVar[linIdx]
			if (varStr == "pressure"):
				field = sol.solPrim[:,0]
			elif (varStr == "velocity"):
				field = sol.solPrim[:,1]
			elif (varStr == "temperature"):
				field = sol.solPrim[:,2]
			elif (varStr == "species"):
				field = sol.solPrim[:,3]
			elif (varStr == "source"):
				field = sol.source[:,0]
			elif (varStr == "density"):
				field = sol.solCons[:,0]
			elif (varStr == "momentum"):
				field = sol.solCons[:,1]
			elif (varStr == "energy"):
				field = sol.solCons[:,2]
			else:
				raise ValueError("Invalid field visualization variable:"+str(varStr))

			axVar.plot(geom.xCell, field)
			axVar.set_ylim(params.visYBounds[linIdx])
			axVar.set_xlim(params.visXBounds[linIdx])
			axVar.set_ylabel(axLabels[linIdx])
			axVar.set_xlabel("x (m)")
			axVar.ticklabel_format(useOffset=False)

	fig.tight_layout()
	plt.show(block=False)
	plt.pause(0.001)

# write field plot image to disk
def writeFieldImg(fig: plt.Figure, params: parameters, tStep, fieldImgDir):

	visIdx 	= int((tStep+1) / params.visInterval)
	figNum 	= params.imgString % visIdx
	figFile = os.path.join(fieldImgDir, "fig_"+figNum+".png")
	fig.savefig(figFile)

# update probeVals, as this happens every time iteration
def updateProbe(sol: solutionPhys, params: parameters, bounds: boundaries, probeVals, probeIdx, tStep):

	for visIdx in range(params.numVis):
		varStr = params.visVar[visIdx]
		if (params.probeSec == "inlet"):
			solPrimProbe = bounds.inlet.sol.solPrim[0,:]
			solConsProbe = bounds.inlet.sol.solCons[0,:]

		elif (params.probeSec == "outlet"):
			solPrimProbe = bounds.outlet.sol.solPrim[0,:]
			solConsProbe = bounds.outlet.sol.solCons[0,:]

		else:
			solPrimProbe = sol.solPrim[probeIdx,:]
			solConsProbe = sol.solCons[probeIdx,:]
			solSourceProbe = sol.source[probeIdx,:]

		try:
			if (varStr == "pressure"):
				probe = solPrimProbe[0]
			elif (varStr == "velocity"):
				probe = solPrimProbe[1]
			elif (varStr == "temperature"):
				probe = solPrimProbe[2]
			elif (varStr == "species"):
				probe = solPrimProbe[3]
			elif (varStr == "source"):
				probe = solSourceProbe[0]
			elif (varStr == "density"):
				probe = solConsProbe[0]
			elif (varStr == "momentum"):
				probe = solConsProbe[1]
			elif (varStr == "energy"):
				probe = solConsProbe[2]
		except:
			raise ValueError("Invalid field visualization variable "+str(params.visVar)+" for probe at "+params.probeSec)
		

		probeVals[tStep, visIdx] = probe 

# plot probeVals at specied visInterval
def plotProbe(fig: plt.Figure, ax: plt.Axes, axLabels, sol: solutionPhys, params: parameters, probeVals, tStep, tVals):

	if (type(ax) != np.ndarray): 
		axList = [ax]
	else:
		axList = ax 

	for colIdx, col in enumerate(axList):
		if (type(col) != np.ndarray):
			colList = [col]
		else:
			colList = col
		for rowIdx, axVar in enumerate(colList):

			axVar.cla()
			linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (params.visNRows, params.visNCols))[0]
			if ((linIdx+1) > params.numVis): 
				axVar.axis("off")
				break

			axVar.plot(tVals[:tStep+1], probeVals[:tStep+1, linIdx])
			axVar.set_ylim(params.visYBounds[linIdx])
			axVar.set_xlim(params.visXBounds[linIdx])
			axVar.set_ylabel(axLabels[linIdx])
			axVar.set_xlabel("t (s)")
			axVar.ticklabel_format(axis='both',useOffset=False)
			axVar.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

	fig.tight_layout()
	plt.show(block=False)
	plt.pause(0.001)

# write snapshot matrices and point monitors to disk
def writeDataUnsteady(sol: solutionPhys, params: parameters, probeVals, tVals):

	# save snapshot matrices to disk
	if params.primOut:
		solPrimFile = os.path.join(params.unsOutDir, "solPrim_"+params.simType+".npy")
		np.save(solPrimFile, sol.primSnap)
	if params.consOut:
		solConsFile = os.path.join(params.unsOutDir, "solCons_"+params.simType+".npy")
		np.save(solConsFile, sol.consSnap) 
	if params.sourceOut:
		sourceFile = os.path.join(params.unsOutDir, "source_"+params.simType+".npy")
		np.save(sourceFile, sol.sourceSnap)
	if params.RHSOut:
		solRHSFile = os.path.join(params.unsOutDir, "solRHS_"+params.simType+".npy")
		np.save(solRHSFile, sol.RHSSnap) 

	# save point monitors to disk
	probeFileName = "probe"
	for visVar in params.visVar:
		probeFileName += "_"+visVar
	probeFile = os.path.join(params.probeOutDir, probeFileName+"_"+params.simType+".npy")
	probeSave = np.concatenate((tVals.reshape(-1,1), probeVals.reshape(-1,params.numVis)), axis=1) 	# TODO: add third reshape dimensions for multiple probes
	np.save(probeFile, probeSave)

# update residual output and "steady" solution
def writeDataSteady(sol: solutionPhys, params: parameters):

	# write field data
	solPrimFile = os.path.join(params.unsOutDir, "solPrim_steady.npy")
	np.save(solPrimFile, sol.solPrim)
	solConsFile = os.path.join(params.unsOutDir, "solCons_steady.npy")
	np.save(solConsFile, sol.solCons)

# append residual to residual file
def updateResOut(sol: solutionPhys, params: parameters, tStep):

	resFile = os.path.join(params.unsOutDir, "steadyResOut.dat")
	if (tStep == 0):
		f = open(resFile,"w")
	else:
		f = open(resFile, "a")
	f.write(str(tStep+1)+"\t"+str(sol.resOutL2)+"\t"+str(sol.resOutL1)+"\n")
	f.close()

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
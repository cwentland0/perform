import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
import os
from solution import solutionPhys, boundaries
import constants
import time
import pdb

mpl.rc('font', family='serif',size='10')
mpl.rc('axes', labelsize='x-large')
mpl.rc('figure', facecolor='w')
mpl.rc('text', usetex=False)
mpl.rc('text.latex',preamble=r'\usepackage{amsmath}')

# TODO: make visualization classes for each type of plot
# TODO: plot/save multiple variables for probe and field/probe images
# TODO: add visualization for arbitrary species
# TODO: move label selection to parameters
# TODO: better automatic formatting of images? Ultimately real plotting will be un utils, might now be worth the time

# store snapshots of field data
def storeFieldDataUnsteady(sol: solutionPhys, solver, tStep):

	storeIdx = int(tStep / solver.outInterval) + 1

	if solver.primOut: sol.primSnap[:,:,storeIdx] = sol.solPrim
	if solver.consOut: sol.consSnap[:,:,storeIdx] = sol.solCons
	if solver.RHSOut:  sol.RHSSnap[:,:,storeIdx]  = sol.RHS

# get figure and axes handles for visualization
# TODO: some way to generalize this for any number of visualization plots?
# 	not even 9 variables to visualize right now
def setupPlotAxes(solver):

	if (solver.numVis == 1):
		solver.visNRows = 1 
		solver.visNCols = 1
	elif (solver.numVis == 2):
		solver.visNRows = 2
		solver.visNCols = 1
	elif (solver.numVis <= 4):
		solver.visNRows = 2
		solver.visNCols = 2
	elif (solver.numVis <= 6):
		solver.visNRows = 3
		solver.visNCols = 2
	elif (solver.numVis <= 9):
		solver.visNRows = 3
		solver.visNCols = 3

	# axis labels
	axLabels = []
	for axIdx in range(solver.numVis):
		varStr = solver.visVar[axIdx]
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
			raise ValueError("Invalid field visualization variable:"+str(solver.visVar))

	fig, ax = plt.subplots(nrows=solver.visNRows, ncols=solver.visNCols, figsize=(12,6))

	return fig, ax, axLabels

# plot field line plots
def plotField(fig: plt.Figure, ax: plt.Axes, axLabels, sol: solutionPhys, solver):

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
			linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (solver.visNRows, solver.visNCols))[0]
			if ((linIdx+1) > solver.numVis): 
				axVar.axis("off")
				break

			varStr = solver.visVar[linIdx]
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

			axVar.plot(solver.mesh.xCell, field)
			axVar.set_ylim(solver.visYBounds[linIdx])
			axVar.set_xlim(solver.visXBounds[linIdx])
			axVar.set_ylabel(axLabels[linIdx])
			axVar.set_xlabel("x (m)")
			axVar.ticklabel_format(useOffset=False)

	fig.tight_layout()
	plt.show(block=False)
	plt.pause(0.001)

# write field plot image to disk
def writeFieldImg(fig: plt.Figure, solver, tStep, fieldImgDir):

	visIdx 	= int((tStep+1) / solver.visInterval)
	figNum 	= solver.imgString % visIdx
	figFile = os.path.join(fieldImgDir, "fig_"+figNum+".png")
	fig.savefig(figFile)

# update probeVals, as this happens every time iteration
def updateProbe(sol: solutionPhys, solver, bounds: boundaries, probeVals, probeIdx, tStep):

	for visIdx in range(solver.numVis):
		varStr = solver.visVar[visIdx]
		if (solver.probeSec == "inlet"):
			solPrimProbe = bounds.inlet.sol.solPrim[0,:]
			solConsProbe = bounds.inlet.sol.solCons[0,:]

		elif (solver.probeSec == "outlet"):
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
			raise ValueError("Invalid field visualization variable "+str(solver.visVar)+" for probe at "+solver.probeSec)
		

		probeVals[tStep, visIdx] = probe 

# plot probeVals at specied visInterval
def plotProbe(fig: plt.Figure, ax: plt.Axes, axLabels, sol: solutionPhys, solver, probeVals, tStep, tVals):

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
			linIdx = np.ravel_multi_index(([colIdx],[rowIdx]), (solver.visNRows, solver.visNCols))[0]
			if ((linIdx+1) > solver.numVis): 
				axVar.axis("off")
				break

			axVar.plot(tVals[:tStep+1], probeVals[:tStep+1, linIdx])
			axVar.set_ylim(solver.visYBounds[linIdx])
			axVar.set_xlim(solver.visXBounds[linIdx])
			axVar.set_ylabel(axLabels[linIdx])
			axVar.set_xlabel("t (s)")
			axVar.ticklabel_format(axis='both',useOffset=False)
			axVar.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

	fig.tight_layout()
	plt.show(block=False)
	plt.pause(0.001)

# write snapshot matrices and point monitors to disk
def writeDataUnsteady(sol: solutionPhys, solver, probeVals, tVals):

	# save snapshot matrices to disk
	if solver.primOut:
		solPrimFile = os.path.join(constants.unsteadyOutputDir, "solPrim_"+solver.simType+".npy")
		np.save(solPrimFile, sol.primSnap)
	if solver.consOut:
		solConsFile = os.path.join(constants.unsteadyOutputDir, "solCons_"+solver.simType+".npy")
		np.save(solConsFile, sol.consSnap) 
	if solver.sourceOut:
		sourceFile = os.path.join(constants.unsteadyOutputDir, "source_"+solver.simType+".npy")
		np.save(sourceFile, sol.sourceSnap)
	if solver.RHSOut:
		solRHSFile = os.path.join(constants.unsteadyOutputDir, "solRHS_"+solver.simType+".npy")
		np.save(solRHSFile, sol.RHSSnap) 

	# save point monitors to disk
	probeFileName = "probe"
	for visVar in solver.visVar:
		probeFileName += "_"+visVar
	probeFile = os.path.join(constants.probeOutputDir, probeFileName+"_"+solver.simType+".npy")
	probeSave = np.concatenate((tVals.reshape(-1,1), probeVals.reshape(-1,solver.numVis)), axis=1) 	# TODO: add third reshape dimensions for multiple probes
	np.save(probeFile, probeSave)

# update residual output and "steady" solution
def writeDataSteady(sol: solutionPhys, solver):

	# write field data
	solPrimFile = os.path.join(constants.unsteadyOutputDir, "solPrim_steady.npy")
	np.save(solPrimFile, sol.solPrim)
	solConsFile = os.path.join(constants.unsteadyOutputDir, "solCons_steady.npy")
	np.save(solConsFile, sol.solCons)

# append residual to residual file
def updateResOut(sol: solutionPhys, solver, tStep):

	resFile = os.path.join(constants.unsteadyOutputDir, "steadyResOut.dat")
	if (tStep == 0):
		f = open(resFile,"w")
	else:
		f = open(resFile, "a")
	f.write(str(tStep+1)+"\t"+str(sol.resOutL2)+"\t"+str(sol.resOutL1)+"\n")
	f.close()

# write restart files containing primitive and conservative fields, plus physical time 
def writeRestartFile(sol: solutionPhys, solver, tStep):

	# write restart file to zipped file
	restartFile = os.path.join(constants.restartOutputDir, "restartFile_"+str(solver.restartIter)+".npz")
	np.savez(restartFile, solTime = solver.solTime, solPrim = sol.solPrim, solCons = sol.solCons)

	# write iteration number files
	restartIterFile = os.path.join(constants.restartOutputDir, "restartIter.dat")
	with open(restartIterFile, "w") as f:
		f.write(str(solver.restartIter)+"\n")

	restartPhysIterFile = os.path.join(constants.restartOutputDir, "restartIter_"+str(solver.restartIter)+".dat")
	with open(restartPhysIterFile, "w") as f:
		f.write(str(tStep+1)+"\n")

	# iterate file count
	if (solver.restartIter < solver.numRestarts):
		solver.restartIter += 1
	else:
		solver.restartIter = 1
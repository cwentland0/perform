import pygems1d.constants as const
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
import os
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
def storeFieldDataUnsteady(solInt, solver):

	storeIdx = int((solver.timeIntegrator.iter - 1) / solver.outInterval) + 1

	if solver.primOut: solInt.primSnap[:,:,storeIdx] = solInt.solPrim
	if solver.consOut: solInt.consSnap[:,:,storeIdx] = solInt.solCons
	if solver.RHSOut:  solInt.RHSSnap[:,:,storeIdx]  = solInt.RHS

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
def plotField(fig: plt.Figure, ax: plt.Axes, axLabels, solInt, solver):

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
				field = solInt.solPrim[:,0]
			elif (varStr == "velocity"):
				field = solInt.solPrim[:,1]
			elif (varStr == "temperature"):
				field = solInt.solPrim[:,2]
			elif (varStr == "species"):
				field = solInt.solPrim[:,3]
			elif (varStr == "source"):
				field = solInt.source[:,0]
			elif (varStr == "density"):
				field = solInt.solCons[:,0]
			elif (varStr == "momentum"):
				field = solInt.solCons[:,1]
			elif (varStr == "energy"):
				field = solInt.solCons[:,2]
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
def writeFieldImg(fig: plt.Figure, solver, fieldImgDir):

	visIdx 	= int(solver.timeIntegrator.iter / solver.visInterval)
	figNum 	= solver.imgString % visIdx
	figFile = os.path.join(fieldImgDir, "fig_"+figNum+".png")
	fig.savefig(figFile)

# update probeVals, as this happens every time iteration
def updateProbe(solDomain, solver, probeVals, probeIdx):

	for visIdx in range(solver.numVis):
		varStr = solver.visVar[visIdx]
		if (solver.probeSec == "inlet"):
			solPrimProbe = solDomain.solIn.solPrim[0,:]
			solConsProbe = solDomain.solIn.solCons[0,:]

		elif (solver.probeSec == "outlet"):
			solPrimProbe = solDomain.solOut.solPrim[0,:]
			solConsProbe = solDomain.solOut.solCons[0,:]

		else:
			solPrimProbe = solDomain.solInt.solPrim[probeIdx,:]
			solConsProbe = solDomain.solInt.solCons[probeIdx,:]
			solSourceProbe = solDomain.solInt.source[probeIdx,:]

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
		

		probeVals[solver.timeIntegrator.iter-1, visIdx] = probe 

# plot probeVals at specied visInterval
def plotProbe(fig: plt.Figure, ax: plt.Axes, axLabels, solver, probeVals, tVals):

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

			axVar.plot(tVals[:solver.timeIntegrator.iter], probeVals[:solver.timeIntegrator.iter, linIdx])
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
def writeDataUnsteady(solInt, solver, probeVals, tVals):

	# save snapshot matrices to disk
	if solver.primOut:
		solPrimFile = os.path.join(const.unsteadyOutputDir, "solPrim_"+solver.simType+".npy")
		np.save(solPrimFile, solInt.primSnap)
	if solver.consOut:
		solConsFile = os.path.join(const.unsteadyOutputDir, "solCons_"+solver.simType+".npy")
		np.save(solConsFile, solInt.consSnap) 
	if solver.sourceOut:
		sourceFile = os.path.join(const.unsteadyOutputDir, "source_"+solver.simType+".npy")
		np.save(sourceFile, solInt.sourceSnap)
	if solver.RHSOut:
		solRHSFile = os.path.join(const.unsteadyOutputDir, "solRHS_"+solver.simType+".npy")
		np.save(solRHSFile, solInt.RHSSnap) 

	# save point monitors to disk
	probeFileName = "probe"
	for visVar in solver.visVar:
		probeFileName += "_"+visVar
	probeFile = os.path.join(const.probeOutputDir, probeFileName+"_"+solver.simType+".npy")
	probeSave = np.concatenate((tVals.reshape(-1,1), probeVals.reshape(-1,solver.numVis)), axis=1) 	# TODO: add third reshape dimensions for multiple probes
	np.save(probeFile, probeSave)

# update residual output and "steady" solution
def writeDataSteady(solInt, solver):

	# write field data
	solPrimFile = os.path.join(const.unsteadyOutputDir, "solPrim_steady.npy")
	np.save(solPrimFile, solInt.solPrim)
	solConsFile = os.path.join(const.unsteadyOutputDir, "solCons_steady.npy")
	np.save(solConsFile, solInt.solCons)

# append residual to residual file
def updateResOut(solInt, solver):

	resFile = os.path.join(const.unsteadyOutputDir, "steadyResOut.dat")
	if (solver.timeIntegrator.iter == 1):
		f = open(resFile,"w")
	else:
		f = open(resFile, "a")
	f.write(str(solver.timeIntegrator.iter)+"\t"+str(solInt.resNormL2)+"\t"+str(solInt.resNormL1)+"\n")
	f.close()

# write restart files containing primitive and conservative fields, plus physical time 
def writeRestartFile(solInt, solver):

	# write restart file to zipped file
	restartFile = os.path.join(const.restartOutputDir, "restartFile_"+str(solver.restartIter)+".npz")
	np.savez(restartFile, solTime = solver.solTime, solPrim = solInt.solPrim, solCons = solInt.solCons)

	# write iteration number files
	restartIterFile = os.path.join(const.restartOutputDir, "restartIter.dat")
	with open(restartIterFile, "w") as f:
		f.write(str(solver.restartIter)+"\n")

	restartPhysIterFile = os.path.join(const.restartOutputDir, "restartIter_"+str(solver.restartIter)+".dat")
	with open(restartPhysIterFile, "w") as f:
		f.write(str(solver.timeIntegrator.iter)+"\n")

	# iterate file count
	if (solver.restartIter < solver.numRestarts):
		solver.restartIter += 1
	else:
		solver.restartIter = 1
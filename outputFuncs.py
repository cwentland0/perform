import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec
import os
from classDefs import parameters, geometry
from solution import solutionPhys
import time
import pdb

# TODO: could go overboard an make a visualization class
# TODO: plot multiple variables with subplots
# TODO: add option to fix y-axis bounds?
# TODO: add visualization for arbitrary species
# TODO: move label selection to parameters

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
	plt.show(block=False)
	plt.pause(0.001)

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
def plotProbe(fig: plt.Figure, ax: plt.Axes, sol: solutionPhys, params: parameters, probeVals, tStep, tVals):

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
	plt.show(block=False)
	plt.pause(0.001)
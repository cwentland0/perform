import numpy as np 
from stateFuncs import calcStateFromPrim, calcGammaMixture 
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries, boundary
import pdb

# TODO: more descriptive variable naming
# @profile
def updateGhostCells(sol, bounds: boundaries, params, geom, gas):

	calcInletGhostCell(sol, bounds.inlet, params, geom, gas)
	calcOutletGhostCell(sol, bounds.outlet, params, geom, gas)

	# update thermo properties
	bounds.inlet.sol.updateState(gas, fromCons=False)
	bounds.outlet.sol.updateState(gas, fromCons=False)

# @profile
def calcInletGhostCell(sol: solutionPhys, inlet: boundary, params: parameters, geom: geometry, gas: gasProps):

	# TODO: this is absolutely not correct
	if (inlet.type == "characteristic"):
		Qp = 2.0 * sol.solPrim[[0],:] - sol.solPrim[[1],:]
		Q, RMix, enthRefMix, CpMix = calcStateFromPrim(Qp, gas)

		J, rhoc = calcCharacteristic(Q, Qp, RMix, CpMix)
		J = -J 

		# CHECK THIS
		# Qp[0,0] = rhoc * (J - Qp[[0],[2]])
		# Q, _, _, _ = calcStateFromPrim(Qp, gas)

		inlet.sol.solPrim = Qp 
		inlet.sol.solCons = Q 

# @profile
def calcOutletGhostCell(sol: solutionPhys, outlet: boundary, params: parameters, geom: geometry, gas: gasProps):

	if (outlet.type == "subsonic"):

		pressBound = outlet.press
		if (outlet.pertType == "pressure"):
			pressBound *= (1.0 + outlet.calcPert(params.solTime))

		# compute interior state
		gamma 		= calcGammaMixture(sol.RMix[-1], sol.CpMix[-1])
		pressPlus 	= sol.solPrim[-1, 0]
		rhoPlus 	= sol.solCons[-1, 0]
		velPlus 	= sol.solPrim[-1, 1]
		SPlus 		= pressPlus / np.power(rhoPlus, gamma)
		cPlus		= np.sqrt(gamma * sol.RMix[-1] * sol.solPrim[-1,2])
		JPlus		= velPlus + 2.0 * cPlus / (gamma - 1.0)

		# compute exterior state
		rhoBound 	= np.power( (pressBound / SPlus), (1.0/gamma) )
		cBound 		= np.sqrt(gamma * pressBound / rhoBound)
		velBound 	= JPlus - 2.0 * cBound / (gamma - 1.0)
		tempBound 	= pressBound / (sol.RMix[-1] * rhoBound)

		# assign ghost cell state
		outlet.sol.solPrim = sol.solPrim[[-1],:].copy()
		outlet.sol.solPrim[0,0] = pressBound 
		outlet.sol.solPrim[0,1] = velBound
		outlet.sol.solPrim[0,2] = tempBound 
		outlet.sol.updateState(gas, fromCons = False)

def calcCharacteristic(solCons, solPrim, RMix, CpMix):

	gamma = calcGammaMixture(RMix, CpMix)
	soundSpeed = np.sqrt(gamma * solPrim[:,0] / solCons[:,0])
	rhoC = solCons[:,0] * soundSpeed
	J = -solPrim[:,0] / rhoC + solPrim[:,1]

	return J, rhoC

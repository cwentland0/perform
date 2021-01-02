import numpy as np 
from math import pow, sqrt
import stateFuncs
from stateFuncs import calcStateFromPrim, calcGammaMixture, calcGasConstantMixture, calcCpMixture
from solution import solutionPhys, boundaries, boundary
from constants import realType
import pdb


# compute boundary ghost cell state
def calcBoundaries(sol, bounds: boundaries, solver):

	calcInlet(sol, bounds.inlet, solver)
	calcOutlet(sol, bounds.outlet, solver)


# compute inlet flux/state
def calcInlet(sol: solutionPhys, inlet: boundary, solver):

	# specify stagnation temperature and stagnation pressure
	# this assumes negligible change in chemical composition across the boundary
	if (inlet.type == "stagnation"):

		# chemical composition assumed constant near boundary
		RMix = inlet.RMix[0]
		gamma = inlet.gamma[0]
		gammaM1 = gamma - 1.0

		# interior state
		velP1 	= sol.solPrim[0, 1]
		velP2 	= sol.solPrim[1, 1]
		cP1 	= sqrt(gamma * RMix * sol.solPrim[0, 2])
		cP2 	= sqrt(gamma * RMix * sol.solPrim[1, 2])

		# interpolate outgoing Riemann invariant
		# negative sign on velocity is to account for flux/boundary normal directions
		J1 = -velP1 - (2.0 * cP1) / gammaM1
		J2 = -velP2 - (2.0 * cP2) / gammaM1

		# extrapolate to exterior
		if (solver.spaceOrder == 1):
			J = J1
		elif (solver.spaceOrder == 2):
			J  = 2.0 * J1 - J2
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

		# quadratic form for exterior Mach number
		c2 	 = gamma * RMix * inlet.temp
		
		aVal = c2 - J**2 * gammaM1 / 2.0
		bVal = (4.0 * c2) / gammaM1
		cVal = (4.0 * c2) / gammaM1**2 - J**2
		rad = bVal**2 - 4.0 * aVal * cVal 

		# check for non-physical solution (usually caused by reverse flow)
		if (rad < 0.0):
			print("aVal: "+str(aVal))
			print("bVal: "+str(bVal))
			print("cVal: "+str(cVal))
			print("Boundary velocity: "+str(velP1))
			raise ValueError("Non-physical inlet state")

		# solve quadratic formula, assign Mach number depending on sign/magnitude 
		# if only one positive, select that. If both positive, select smaller
		rad = sqrt(rad)
		mach1 = (-bVal - rad) / (2.0 * aVal) 
		mach2 = (-bVal + rad) / (2.0 * aVal)
		if ((mach1 > 0) and (mach2 > 0)):
			machBound = min(mach1, mach2)
		elif ((mach1 <= 0) and (mach2 <=0)):
			raise ValueError("Non-physical Mach number at inlet")
		else:
			machBound = max(mach1, mach2)

		# compute exterior state
		tempBound 	= inlet.temp / (1.0 +  gammaM1 / 2.0 * machBound**2) 
		pressBound 	= inlet.press * pow(tempBound / inlet.temp, gamma / gammaM1) 
		cBound 		= sqrt(gamma * RMix * tempBound)
		velBound 	= machBound * cBound

		# calculate ghost cell state
		inlet.sol.solPrim[0,0] = pressBound
		inlet.sol.solPrim[0,1] = velBound
		inlet.sol.solPrim[0,2] = tempBound
		inlet.sol.updateState(solver.gasModel, fromCons = False)

	# full state specification
	# mostly just for perturbing inlet state to check for outlet reflections
	elif (inlet.type == "fullstate"):

		pressBound 	= inlet.press
		velBound 	= inlet.vel
		tempBound 	= inlet.temp

		# perturbation
		if (inlet.pertType == "velocity"):
			velBound *= (1.0 + inlet.calcPert(solver.solTime))
		elif (inlet.pertType == "pressure"):
			pressBound *= (1.0 + inlet.calcPert(solver.solTime))

		# compute ghost cell state
		inlet.sol.solPrim[0,0] = pressBound
		inlet.sol.solPrim[0,1] = velBound
		inlet.sol.solPrim[0,2] = tempBound
		inlet.sol.updateState(solver.gasModel, fromCons = False)

	# non-reflective boundary, unsteady solution is perturbation about mean flow solution
	# refer to documentation for derivation
	elif (inlet.type == "meanflow"):

		# mean flow and infinitely-far upstream quantities
		pressUp 	= inlet.press 
		tempUp 		= inlet.temp
		massFracUp	= inlet.massFrac[:-1]
		rhoCMean 	= inlet.vel 
		rhoCpMean 	= inlet.rho

		if (inlet.pertType == "pressure"):
			pressUp *= (1.0 + inlet.calcPert(solver.solTime))

		# interior quantities
		pressIn 	= sol.solPrim[:2,0]
		velIn 		= sol.solPrim[:2,1]

		# pdb.set_trace()
		# characteristic variables
		w3In 	= velIn - pressIn / rhoCMean  

		# extrapolate to exterior
		if (solver.spaceOrder == 1):
			w3Bound = w3In[0]
		elif (solver.spaceOrder == 2):
			w3Bound = 2.0*w3In[0] - w3In[1]
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

		# compute exterior state
		pressBound 	= (pressUp - w3Bound * rhoCMean) / 2.0
		velBound 	= (pressUp - pressBound) / rhoCMean 
		tempBound 	= tempUp + (pressBound - pressUp) / rhoCpMean
		massFracBound = massFracUp 

		# set ghost cell state
		inlet.sol.solPrim[0,0] 	= pressBound
		inlet.sol.solPrim[0,1] 	= velBound
		inlet.sol.solPrim[0,2] 	= tempBound
		inlet.sol.solPrim[0,3:] = massFracBound
		inlet.sol.updateState(solver.gasModel, fromCons = False)

	elif (inlet.type == "specTU"):
		tempUp 		= inlet.temp
		velUp 		= inlet.vel 
		massFracUp	= inlet.massFrac[:-1]
		rhoCMean 	= inlet.rho 

		pressIn 	= sol.solPrim[:2,0]
		velIn 		= sol.solPrim[:2,1]

		w3In 	= velIn - pressIn / rhoCMean 
		if (solver.spaceOrder == 1):
			w3Bound = w3In[0]
		elif (solver.spaceOrder == 2):
			w3Bound = 2.0*w3In[0] - w3In[1]
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

		pressBound = rhoCMean * (velUp - w3Bound)
		
		inlet.sol.solPrim[0,0] = pressBound 
		inlet.sol.solPrim[0,1] = velUp 
		inlet.sol.solPrim[0,2] = tempUp 
		inlet.sol.solPrim[0,3] = massFracUp
		inlet.sol.updateState(solver.gasModel, fromCons = False)

	else:
		raise ValueError("Invalid inlet boundary condition choice: "+inlet.type)

# compute outlet flux/state
def calcOutlet(sol: solutionPhys, outlet: boundary, solver):

	# subsonic outflow, specify outlet static pressure
	# this assumes negligible change in chemical composition across boundary
	if (outlet.type == "subsonic"):

		pressBound = outlet.press
		if (outlet.pertType == "pressure"):
			pressBound *= (1.0 + outlet.calcPert(solver.solTime))

		# chemical composition assumed constant near boundary
		RMix 		= outlet.RMix[0]
		gamma 		= outlet.gamma[0]
		gammaM1 	= gamma - 1.0

		# calculate interior state
		pressP1 	= sol.solPrim[-1, 0]
		pressP2 	= sol.solPrim[-2, 0]
		rhoP1 		= sol.solCons[-1, 0]
		rhoP2 		= sol.solCons[-2, 0]
		velP1 		= sol.solPrim[-1, 1]
		velP2 		= sol.solPrim[-2, 1]

		# outgoing characteristics information
		SP1 		= pressP1 / pow(rhoP1, gamma) 				# entropy
		SP2 		= pressP2 / pow(rhoP2, gamma)
		cP1			= sqrt(gamma * RMix * sol.solPrim[-1,2]) 	# sound speed
		cP2			= sqrt(gamma * RMix * sol.solPrim[-2,2])
		JP1			= velP1 + 2.0 * cP1 / gammaM1				# u+c Riemann invariant
		JP2			= velP2 + 2.0 * cP2 / gammaM1
		
		# extrapolate to exterior
		if (solver.spaceOrder == 1):
			S = SP1
			J = JP1
		elif (solver.spaceOrder == 2):
			S = 2.0 * SP1 - SP2
			J = 2.0 * JP1 - JP2
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

		# compute exterior state
		rhoBound 	= pow( (pressBound / S), (1.0/gamma) )
		cBound 		= sqrt(gamma * pressBound / rhoBound)
		velBound 	= J - 2.0 * cBound / gammaM1
		tempBound 	= pressBound / (RMix * rhoBound)		

		# calculate ghost cell state
		outlet.sol.solPrim[0,0] = pressBound
		outlet.sol.solPrim[0,1] = velBound
		outlet.sol.solPrim[0,2] = tempBound
		outlet.sol.updateState(solver.gasModel, fromCons = False)

	# non-reflective boundary, unsteady solution is perturbation about mean flow solution
	# refer to documentation for derivation
	elif (outlet.type == "meanflow"):

		# specify rho*C and rho*Cp from mean solution, back pressure is static pressure at infinity
		rhoCMean 	= outlet.vel 
		rhoCpMean 	= outlet.rho
		pressBack 	= outlet.press 

		if (outlet.pertType == "pressure"):
			pressBack *= (1.0 + outlet.calcPert(solver.solTime))

		# interior quantities
		pressOut 	= sol.solPrim[-2:,0]
		velOut 		= sol.solPrim[-2:,1]
		tempOut 	= sol.solPrim[-2:,2]
		massFracOut = sol.solPrim[-2:,3:]

		# characteristic variables
		w1Out 	= tempOut - pressOut / rhoCpMean
		w2Out 	= velOut + pressOut / rhoCMean
		w4Out 	= massFracOut 

		# extrapolate to exterior
		if (solver.spaceOrder == 1):
			w1Bound = w1Out[0]
			w2Bound = w2Out[0]
			w4Bound = w4Out[0,:]
		elif (solver.spaceOrder == 2):
			w1Bound = 2.0*w1Out[0] - w1Out[1]
			w2Bound = 2.0*w2Out[0] - w2Out[1]
			w4Bound = 2.0*w4Out[0,:] - w4Out[1,:]
		else:
			raise ValueError("Higher order extrapolation implementation required for spatial order "+str(solver.spaceOrder))

		# compute exterior state
		pressBound 	= (w2Bound * rhoCMean + pressBack) / 2.0
		velBound 	= (pressBound - pressBack) / rhoCMean 
		tempBound 	= w1Bound + pressBound / rhoCpMean 
		massFracBound = w4Bound 

		# set ghost cell state
		outlet.sol.solPrim[0,0] = pressBound
		outlet.sol.solPrim[0,1] = velBound
		outlet.sol.solPrim[0,2] = tempBound
		outlet.sol.solPrim[0,3:] = massFracBound
		outlet.sol.updateState(solver.gasModel, fromCons = False)
			

	else:
		raise ValueError("Invalid outlet boundary condition choice: "+outler.type)
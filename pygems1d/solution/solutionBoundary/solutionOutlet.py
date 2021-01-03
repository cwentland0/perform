from .solutionBoundary import solutionBoundary
from inputFuncs import parseBC
from math import pow, sqrt

class solutionOutlet(solutionBoundary):
	"""
	Outlet ghost cell solution
	"""

	def __init__(self, solver):

		paramDict = solver.paramDict
		self.boundCond 		= paramDict["boundCond_outlet"] 

		# add assertions to check that required properties are specified
		if (self.boundCond == "subsonic"):
			self.boundFunc = self.calcSubsonicBC
		elif (self.boundCond == "meanflow"):
			self.boundFunc = self.calcMeanFlowBC
		else:
			raise ValueError("Invalid outlet boundary condition selection: " + str(self.boundCond))

		super().__init__(solver, "outlet")

	def calcSubsonicBC(self, solver, solPrim=None, solCons=None):
		"""
		Subsonic outflow, specify outlet static pressure
		"""

		assert ((solPrim is not None) and (solCons is not None)), "Must provide primitive and conservative interior state."

		pressBound = self.press
		if (self.pertType == "pressure"):
			pressBound *= (1.0 + self.calcPert(solver.solTime))

		# chemical composition assumed constant near boundary
		RMix 		= self.RMix[0]
		gamma 		= self.gamma[0]
		gammaM1 	= gamma - 1.0

		# calculate interior state
		pressP1 	= solPrim[-1, 0]
		pressP2 	= solPrim[-2, 0]
		rhoP1 		= solCons[-1, 0]
		rhoP2 		= solCons[-2, 0]
		velP1 		= solPrim[-1, 1]
		velP2 		= solPrim[-2, 1]

		# outgoing characteristics information
		SP1 		= pressP1 / pow(rhoP1, gamma) 				# entropy
		SP2 		= pressP2 / pow(rhoP2, gamma)
		cP1			= sqrt(gamma * RMix * solPrim[-1,2]) 	# sound speed
		cP2			= sqrt(gamma * RMix * solPrim[-2,2])
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
		self.solPrim[0,0] 	= pressBound
		rhoBound 			= pow( (pressBound / S), (1.0/gamma) )
		cBound 				= sqrt(gamma * pressBound / rhoBound)
		self.solPrim[0,1] 	= J - 2.0 * cBound / gammaM1
		self.solPrim[0,2] 	= pressBound / (RMix * rhoBound)		

	def calcMeanFlowBC(self, solver, solPrim=None, solCons=None):
		"""
		Non-reflective boundary, unsteady solution is perturbation about mean flow solution
		Refer to documentation for derivation
		"""

		assert (solPrim is not None), "Must provide primitive interior state"

		# specify rho*C and rho*Cp from mean solution, back pressure is static pressure at infinity
		rhoCMean 	= self.vel 
		rhoCpMean 	= self.rho
		pressBack 	= self.press 

		if (self.pertType == "pressure"):
			pressBack *= (1.0 + self.calcPert(solver.solTime))

		# interior quantities
		pressOut 	= solPrim[-2:,0]
		velOut 		= solPrim[-2:,1]
		tempOut 	= solPrim[-2:,2]
		massFracOut = solPrim[-2:,3:]

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
		pressBound 			= (w2Bound * rhoCMean + pressBack) / 2.0
		self.solPrim[0,0] 	= pressBound
		self.solPrim[0,1] 	= (pressBound - pressBack) / rhoCMean 
		self.solPrim[0,2] 	= w1Bound + pressBound / rhoCpMean 
		self.solPrim[0,3:] 	= w4Bound 
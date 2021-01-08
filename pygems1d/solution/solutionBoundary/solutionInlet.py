from pygems1d.solution.solutionBoundary.solutionBoundary import solutionBoundary

from math import pow, sqrt

class solutionInlet(solutionBoundary):
	"""
	Inlet ghost cell solution
	"""

	def __init__(self, solver):

		paramDict = solver.paramDict
		self.boundCond = paramDict["boundCond_inlet"] 

		# add assertions to check that required properties are specified
		if (self.boundCond == "stagnation"):
			self.boundFunc = self.calcStagnationBC
		elif (self.boundCond == "fullstate"):
			self.boundFunc = self.calcFullStateBC
		elif (self.boundCond == "meanflow"):
			self.boundFunc = self.calcMeanFlowBC
		else:
			raise ValueError("Invalid inlet boundary condition selection: " + str(self.boundCond))

		super().__init__(solver, "inlet")


	def calcStagnationBC(self, solver, solPrim=None, solCons=None):
		"""
		Specify stagnation temperature and stagnation pressure
		"""

		assert (solPrim is not None), "Must provide primitive interior state"

		# chemical composition assumed constant near boundary
		RMix = self.RMix[0]
		gamma = self.gamma[0]
		gammaM1 = gamma - 1.0

		# interior state
		velP1 	= solPrim[1, 0]
		velP2 	= solPrim[1, 1]
		cP1 	= sqrt(gamma * RMix * solPrim[2, 0])
		cP2 	= sqrt(gamma * RMix * solPrim[2, 1])

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
		c2 	 = gamma * RMix * self.temp
		
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
		tempBound 			= self.temp / (1.0 +  gammaM1 / 2.0 * machBound**2) 
		self.solPrim[2,0] 	= tempBound
		self.solPrim[0,0] 	= self.press * pow(tempBound / self.temp, gamma / gammaM1) 
		cBound 				= sqrt(gamma * RMix * tempBound)
		self.solPrim[1,0] 	= machBound * cBound

	def calcFullStateBC(self, solver, solPrim=None, solCons=None):
		"""
		Full state specification
		Mostly just for perturbing inlet state to check for outlet reflections
		"""

		pressBound 	= self.press
		velBound 	= self.vel
		tempBound 	= self.temp

		# perturbation
		
		if (self.pertType == "pressure"):
			pressBound *= (1.0 + self.calcPert(solver.solTime))
		elif (self.pertType == "velocity"):
			velBound *= (1.0 + self.calcPert(solver.solTime))
		elif (self.pertType == "temperature"):
			pressBound *= (1.0 + self.calcPert(solver.solTime))

		# compute ghost cell state
		self.solPrim[0,0] = pressBound
		self.solPrim[1,0] = velBound
		self.solPrim[2,0] = tempBound

	def calcMeanFlowBC(self, solver, solPrim=None, solCons=None):
		"""
		Non-reflective boundary, unsteady solution is perturbation about mean flow solution
		Refer to documentation for derivation
		"""

		assert (solPrim is not None), "Must provide primitive interior state"

		# mean flow and infinitely-far upstream quantities
		pressUp 	= self.press 
		tempUp 		= self.temp
		massFracUp	= self.massFrac[:-1]
		rhoCMean 	= self.vel 
		rhoCpMean 	= self.rho

		if (self.pertType == "pressure"):
			pressUp *= (1.0 + self.calcPert(solver.solTime))

		# interior quantities
		pressIn 	= solPrim[0,:2]
		velIn 		= solPrim[1,:2]

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
		pressBound 			= (pressUp - w3Bound * rhoCMean) / 2.0
		self.solPrim[0,0] 	= pressBound
		self.solPrim[1,0] 	= (pressUp - pressBound) / rhoCMean 
		self.solPrim[2,0] 	= tempUp + (pressBound - pressUp) / rhoCpMean
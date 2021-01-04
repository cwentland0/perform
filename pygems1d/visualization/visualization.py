import matplotlib.pyplot as plt

# TODO: this is kind of a mess
# 	I'm struggling to write effective hierarchy for field, probe, and residual, as probe shares similarities with field and residual plots
# TODO: add RHS, flux plotting

class visualization:

	def __init__(self, visType, solver):

		if (self.numSubplots == 1):
			self.numRows = 1 
			self.numCols = 1
		elif (self.numSubplots == 2):
			self.numRows = 2
			self.numCols = 1
		elif (self.numSubplots <= 4):
			self.numRows = 2
			self.numCols = 2
		elif (self.numSubplots <= 6):
			self.numRows = 3
			self.numCols = 2
		elif (self.numSubplots <= 9):
			self.numRows = 3
			self.numCols = 3
		else:
			raise ValueError("Cannot plot more than nine subplots in the same image")

		# axis labels
		# could change this to a dictionary reference
		self.axLabels = [None] * self.numSubplots
		if (self.visType == "residual"):
			self.axLabels[0] = "Residual History"
		else:
			for axIdx in range(self.numSubplots):
				varStr = self.visVar[axIdx]
				if (varStr == "pressure"):
					self.axLabels[axIdx] = "Pressure (Pa)"
				elif (varStr == "velocity"):
					self.axLabels[axIdx] = "Velocity (m/s)"
				elif (varStr == "temperature"):
					self.axLabels[axIdx] = "Temperature (K)"
				elif (varStr == "source"):
					self.axLabels[axIdx] = "Reaction Source Term"
				elif (varStr == "density"):
					self.axLabels[axIdx] = "Density (kg/m^3)"
				elif (varStr == "momentum"):
					self.axLabels[axIdx] = "Momentum (kg/s-m^2)"
				elif (varStr == "energy"):
					self.axLabels[axIdx] = "Total Energy"

				# TODO: some way to incorporate actual species name
				elif (varStr[:7] == "species"):
					self.axLabels[axIdx] = "Species "+str(varStr[7:])+" Mass Fraction" 
				elif (varStr[:15] == "density-species"):
					self.axLabels[axIdx] = "Density-weighted Species "+str(varStr[7:])+" Mass Fraction (kg/m^3)"
				else:
					raise ValueError("Invalid field visualization variable:"+str(solver.visVar))

		# self.fig, self.ax = plt.subplots(nrows=self.numRows, ncols=self.numCols, num=self.visID)



	def getFieldData(self, solDomain, varStr):
		"""
		Extract plotting data from flow field domain data
		"""

		if (self.visType == "field"):
			solPrim = solDomain.solInt.solPrim
			solCons = solDomain.solInt.solCons
			source  = solDomain.solInt.source
			rhs 	= solDomain.solInt.RHS
		elif (self.visType == "probe"):
			if (self.probeSec == "interior"):
				solPrim = solDomain.solInt.solPrim[[self.probeIdx],:]
				solCons = solDomain.solInt.solCons[[self.probeIdx],:]
				source  = solDomain.solInt.source[[self.probeIdx],:]
				rhs  	= solDomain.solInt.RHS[[self.probeIdx],:]
			elif (self.probeSec == "inlet"):
				solPrim = solDomain.solIn.solPrim[[0],:]
				solCons = solDomain.solIn.solCons[[0],:]
			elif (self.probeSec == "outlet"):
				solPrim = solDomain.solOut.solPrim[[0],:]
				solCons = solDomain.solOut.solCons[[0],:]
			else:
				raise ValueError("Invalid probeSec passed to getFieldData")
		else:
			raise ValueError("Invalid visType was passed to getFieldData")

		if (varStr == "pressure"):
			extrData = solPrim[:,0]
		elif (varStr == "velocity"):
			extrData = solPrim[:,1]
		elif (varStr == "temperature"):
			extrData = solPrim[:,2]
		elif (varStr == "source"):
			extrData = source[:,0]
		elif (varStr == "density"):
			extrData = solCons[:,0]
		elif (varStr == "momentum"):
			extrData = solCons[:,1]
		elif (varStr == "energy"):
			extrData = solCons[:,2]

		# TODO: fails for last species
		elif (varStr[:7] == "species"):
			specIdx = int(varStr[7:])
			extrData = solPrim[:,3+specIdx-1]
		elif (varStr[:15] == "density-species"):
			specIdx = int(varStr[15:])
			extrData = solCons[:,3+specIdx-1]
		else:
			raise ValueError("Invalid field visualization variable:"+str(varStr))

		return extrData
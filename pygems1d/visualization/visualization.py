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
		# TODO: could change this to a dictionary reference
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

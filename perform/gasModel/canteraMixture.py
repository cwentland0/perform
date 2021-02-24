from perform.constants import realType, RUniv, suthTemp
from perform.gasModel.gasModel import gasModel

import numpy as np
import cantera as ct
import pdb

class canteraMixture(gasModel):
	"""
	Container class for all Cantera thermo/transport property methods
	"""

	def __init__(self, gasDict,numCells):

		self.gasType 				= gasDict["gasType"]
		self.gas				=	ct.Solution(gasDict["ctiFile"])
		self.Tmin 				= gasDict["Tmin"]
		self.gas.TP			=   self.Tmin,101325
		#self.gasArray			=	ct.SolutionArray(self.gas,numCells)  #used for keeping track of cell properties
		#self.gasChecker			=   ct.SolutionArray(self.gas,1)  #used for state independent lookup
		
		self.numSpeciesFull 	= self.gas.n_species				# total number of species in case
		self.SpeciesNames       = self.gas.species_names
		self.molWeights 		= self.gas.molecular_weights	# molecular weights, g/mol
		#These are either not constant or not needed for TPG
		#self.enthRef 			= -2.545e+05#gasDict["enthRef"].astype(realType) 		# reference enthalpy, J/kg
		#self.tempRef 			= 0.0#gasDict["tempRef"]						# reference temperature, K
		#self.Cp 				= gasDict["Cp"].astype(realType)			# heat capacity at constant pressure, J/(kg-K)
		#self.Pr 				= gasDict["Pr"].astype(realType)			# Prandtl number
		#self.Sc 				= gasDict["Sc"].astype(realType)			# Schmidt number
		#self.muRef				= gasDict["muRef"].astype(realType)			# reference dynamic viscosity for Sutherland model
		

		#Don't need these for cantera all reactions are handled internally
		#self.nu 				= gasDict["nu"].astype(realType)		# ?????
		#self.nuArr 				= gasDict["nuArr"].astype(realType)		# ?????
		#self.actEnergy			= float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		#self.preExpFact 		= float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		#self.RGas 				= RUniv / self.molWeights 			# specific gas constant of each species, J/(K*kg)
		
		self.numSpecies 		= self.numSpeciesFull - 1			# last species is not directly solved for
		print(self.numSpecies)
		self.numEqs 			= self.numSpecies + 3				# pressure, velocity, temperature, and species transport
		self.massFracSlice  = np.arange(self.numSpecies)
		#self.molWeightNu 		= self.molWeights * self.nu 
		#self.mwInvDiffs 		= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		#self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		#self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]

	def tempLim(self,temp):
		return np.where(temp>self.Tmin,temp,self.Tmin)

	def padMassFrac(self,nm1MassFrac):
		nMassFrac = np.concatenate((nm1MassFrac, (1 - np.sum(nm1MassFrac, axis = 0, keepdims = True))), axis = 0)
		return nMassFrac


	def calcMixGasConstant(self, massFrac):
		gasArray=ct.SolutionArray(self.gas,massFrac.shape[1])
		gasArray.TPY=gasArray.T,gasArray.P, self.padMassFrac(massFrac).transpose()
		return RUniv/gasArray.mean_molecular_weight


	def calcMixGamma(self, RMix, CpMix):
		"""
		Compute mixture ratio of specific heats
		"""
		gammaMix = CpMix / (CpMix - RMix)
		return gammaMix


	def calcEnthalpy(self,density,vel,temp,pressure,Y,enthRefMix,CpMix):
		temp=self.tempLim(temp)
		gasArray=ct.SolutionArray(self.gas,Y.shape[1])
		gasArray.TPY=temp,pressure,self.padMassFrac(Y).transpose()
		#print(gasArray.enthalpy_mass[0],gasArray.Y[0])
		return density * (gasArray.enthalpy_mass + np.power(vel,2.)/2)- pressure

	def calcTemperature(self,rho,rhoU,rhoH,rhoY,enthRefMix,CpMix,RMix):
		# Calculates Temperature from enthalpy
		# Requires convergence and recomputation of cp and temperature
		hSense= (rhoH/rho) - np.square(rhoU/rho)/2 
		gasArray=ct.SolutionArray(self.gas,rho.shape[0])
		#1st guess
		ft=1
		Ti=self.gas.T
		T=gasArray.T
		iter_max=20
		conv_tol=1e-12
		gasArray.TDY= Ti , rho, self.padMassFrac(rhoY/rho).transpose()
		#print(gasArray.Y[0])
		print("Temp","Sense(goal) H","H(calced)","h0","KE","species","temp min")
		for i in range(iter_max):
			gasArray.TDY = T,rho,gasArray.Y
			dh = hSense - gasArray.enthalpy_mass
			cp = self.calcMixCp(rhoY/rho,gasArray.T)
			dT = dh/cp 
			if(np.min(T+dT)<0): #Try to minimize overshoot
				ft=.1
			T=T+ft*dT
			print(T[0],hSense[0],gasArray.enthalpy_mass[0],(rhoH[0]/rho[0]), np.square(rhoU[0]/rho[0])/2, gasArray.Y[0],np.min(T))
			if( np.max(np.abs(dT)/T)<conv_tol):
				break
			
		return gasArray.T

	# compute mixture specific heat at constant pressure
	def calcMixCp(self, massFrac,temperature):
		if(temperature is None):
			print("TPG assuming 300K")
			temperature=300*np.ones(massFrac.shape[1])
		temperature=self.tempLim(temperature)
		gasArray=ct.SolutionArray(self.gas,massFrac.shape[1])
		gasArray.TPY=temperature,gasArray.P, self.padMassFrac(massFrac).transpose()
		return gasArray.cp_mass

	# compute density from ideal gas law  
	def calcDensity(self, solPrim, RMix=None):
		# need to calculate mixture gas constant
		if (RMix is None):
			RMix = self.calcMixGasConstant(solPrim[3:,:])

		# calculate directly from ideal gas
		return  solPrim[0,:] / (RMix * solPrim[2,:])
		

	# compute individual enthalpies for each species
	def calcSpeciesEnthalpies(self, temperature):
		temperature=self.tempLim(temperature)
		gasArray=ct.SolutionArray(self.gas,temperature.size)
		gasArray.TPY=temperature,gasArray.P,gasArray.Y
		speciesEnthalpy=gasArray.partial_molar_enthalpies/gasArray.molecular_weights
		return speciesEnthalpy.transpose()

	# calculate Denisty from Primitive state
	def calcDensityFromPrim(self,solPrim,solCons):
		assert(False)
		return 
	# calculate Momentum from Primitive state
	def calcMomentumFromPrim(self,solPrim,solCons):
		assert(False)
		return 
	# calculate Enthalpy from Primitive state
	def calcEnthalpyFromPrim(self,solPrim,solCons):
		assert(False)
		return 
	# calculate rhoY from Primitive state
	def calcDensityYFromPrim(self,solPrim,solCons):
		assert(False)
		return 


	def calcStagnationEnthalpy(self, solPrim, speciesEnth=None):
		"""
		Compute stagnation enthalpy from velocity and species enthalpies
		"""
		gasArray=ct.SolutionArray(self.gas,solPrim.shape[1])
		gasArray.TPY=solPrim[2,:].transpose(),solPrim[0,:].transpose(),self.padMassFrac(solPrim[3:,:]).transpose()
		

		stagEnth = gasArray.enthalpy_mass + 0.5 * np.square(solPrim[1,:])

		return stagEnth



	def calcDensityDerivatives(self, density, 
								wrtPress=False, pressure=None,
								wrtTemp=False, temperature=None,
								wrtSpec=False, mixMolWeight=None, massFracs=None):

		"""
		Compute derivatives of density with respect to pressure, temperature, or species mass fraction
		For species derivatives, returns numSpecies derivatives
		"""

		assert any([wrtPress, wrtTemp, wrtSpec]), "Must compute at least one density derivative..."

		derivs = tuple()
		if (wrtPress):
			assert (pressure is not None), "Must provide pressure for pressure derivative..."
			DDensDPress = density / pressure
			derivs = derivs + (DDensDPress,)

		if (wrtTemp):
			assert (temperature is not None), "Must provide temperature for temperature derivative..."
			DDensDTemp = -density / temperature
			derivs = derivs + (DDensDTemp,)

		if (wrtSpec):
			# calculate mixture molecular weight
			if (mixMolWeight is None):
				assert (massFracs is not None), "Must provide mass fractions to calculate mixture mol weight..."
				mixMolWeight = self.calcMixMolWeight(massFracs)

			DDensDSpec = np.zeros((self.numSpecies, density.shape[0]), dtype=realType)
			for specNum in range(self.numSpecies):
				DDensDSpec[specNum, :] = density * mixMolWeight * (1.0 / self.molWeights[-1] - 1.0 / self.molWeights[specNum])
			derivs = derivs + (DDensDSpec,)

		return derivs


	def calcStagEnthalpyDerivatives(self, wrtPress=False,
									wrtTemp=False, massFracs=None,
									wrtVel=False, velocity=None,
									wrtSpec=False, speciesEnth=None, temperature=None):

		"""
		Compute derivatives of stagnation enthalpy with respect to pressure, temperature, velocity, or species mass fraction
		For species derivatives, returns numSpecies derivatives
		"""

		assert any([wrtPress, wrtTemp, wrtVel, wrtSpec]), "Must compute at least one density derivative..."

		derivs = tuple()
		if (wrtPress):
			DStagEnthDPress = 0.0
			derivs = derivs + (DStagEnthDPress,)
		
		if (wrtTemp):
			assert (massFracs is not None), "Must provide mass fractions for temperature derivative..."

			massFracs = self.getMassFracArray(massFracs=massFracs)
			DStagEnthDTemp = self.calcMixCp(massFracs,temperature)
			derivs = derivs + (DStagEnthDTemp,)

		if (wrtVel):
			assert (velocity is not None), "Must provide velocity for velocity derivative..."
			DStagEnthDVel = velocity.copy()
			derivs = derivs + (DStagEnthDVel,)

		if (wrtSpec):
			if (speciesEnth is None):
				assert (temperature is not None), "Must provide temperature if not providing species enthalpies..."
				speciesEnth = self.calcSpeciesEnthalpies(temperature)
			
			DStagEnthDSpec = np.zeros((self.numSpecies, speciesEnth.shape[1]), dtype=realType)
			if (self.numSpeciesFull == 1):
				DStagEnthDSpec[0,:] = speciesEnth[0,:]
			else:
				for specNum in range(self.numSpecies):
					DStagEnthDSpec[specNum,:] = speciesEnth[specNum,:] - speciesEnth[-1,:]

			derivs = derivs + (DStagEnthDSpec,)

		return derivs


	def calcSoundSpeed(self, temperature, RMix=None, gammaMix=None, massFracs=None, CpMix=None):
		"""
		Compute sound speed
		"""

		# calculate mixture gas constant if not provided
		massFracsSet = False
		if (RMix is None):
			assert (massFracs is not None), "Must provide mass fractions to calculate mixture gas constant..."
			massFracs = self.getMassFracArray(massFracs=massFracs)
			massFracsSet = True
			RMix = self.calcMixGasConstant(massFracs)
		else:
			RMix = np.squeeze(RMix)
			
		# calculate ratio of specific heats if not provided
		if (gammaMix is None):
			if (CpMix is None):
				assert (massFracs is not None), "Must provide mass fractions to calculate mixture Cp..."
				if (not massFracsSet): 
					massFracs = self.getMassFracArray(massFracs=massFracs)
				CpMix = self.calcMixCp(massFracs)
			else:
				CpMix = np.squeeze(CpMix)

			gammaMix = self.calcMixGamma(RMix, CpMix)
		else:
			gammaMix = np.squeeze(gammaMix)

		soundSpeed = np.sqrt(gammaMix * RMix * temperature)

		return soundSpeed

	def calcMixMolWeight(self, massFracs):
		"""
		Compute mixture molecular weight
		"""
		gasArray=ct.SolutionArray(self.gas,massFracs.shape[1])

		if (massFracs.shape[0] == self.numSpecies):
			massFracs = self.padMassFrac(massFracs)

		gasArray.TPY=gasArray.T,gasArray.P,massFracs.transpose()
		mixMolWeight = gasArray.mean_molecular_weight

		return mixMolWeight

	def calcAllMoleFracs(self, massFracs, mixMolWeight=None):
		"""
		Compute mole fractions of all species from mass fractions
		"""
		gasArray=ct.SolutionArray(self.gas,massFracs.shape[1])

		if (massFracs.shape[0] == self.numSpecies):
			massFracs = self.padMassFrac(massFracs)

		gasArray.TPY=gasArray.T,gasArray.P,massFracs.transpose()
		moleFracs = gasArray.X.transpose()

		return moleFracs


	def calcSpeciesDynamicVisc(self, temperature):
		"""
		Compute individual dynamic viscosities from Sutherland's law
		Defaults to reference dynamic viscosity if reference temperature is zero
		Returns values for ALL species, NOT numSpecies species
		"""

		
		return None




	def calcMixDynamicVisc(self, specDynVisc=None, temperature=None, moleFracs=None, massFracs=None):
		"""
		Compute mixture dynamic viscosity from Wilkes mixing law
		"""
		
		assert(temperature is not None), "Must provide temperature"

		if (specDynVisc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
		gasArray=ct.SolutionArray(self.gas,temperature.size)
		if (moleFracs is None):
			assert (massFracs is not None),  "Must provide mass fractions if not providing mole fractions"
			gasArray.TPY=temperature,gasArray.P,massFracs.transpose()
		else:
			gasArray.TPX=temperature,gasArray.P,moleFracs.transpose()
		
		mixDynVisc = gasArray.viscosity

		return mixDynVisc



	def calcMixThermalCond(self, specThermCond=None, specDynVisc=None, temperature=None, moleFracs=None, massFracs=None):
		"""
		Compute mixture thermal conductivity
		"""
		assert(temperature is not None), "Must provide temperature"



		if (specDynVisc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
		gasArray=ct.SolutionArray(self.gas,temperature.size)
		if (moleFracs is None):
			assert (massFracs is not None),  "Must provide mass fractions if not providing mole fractions"
			gasArray.TPY=temperature,gasArray.P,massFracs.transpose()
		else:
			gasArray.TPX=temperature,gasArray.P,moleFracs.transpose()

			mixThermCond = gasArray.thermal_conductivity

		return mixThermCond



	def calcSpeciesMassDiffCoeff(self, density=None, specDynVisc=None, temperature=None,moleFracs=None,massFracs=None):
		"""
		Compute mass diffusivity coefficient of species into mixture
		Returns values for ALL species, NOT numSpecies species
		"""

		if (specDynVisc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			assert (moleFracs is not None), ""
		gasArray=ct.SolutionArray(self.gas,temperature.size)
		if (moleFracs is None):
			assert (massFracs is not None),  "Must provide mass fractions if not providing mole fractions"
			gasArray.TPY=temperature,gasArray.P,massFracs.transpose()
		else:
			gasArray.TPX=temperature,gasArray.P,moleFracs.transpose()
		specMassDiff = gasArray.mix_diff_coeffs.transpose()

		return specMassDiff



	def calcSource(self,temp,massFracs,rho):

		gasArray=ct.SolutionArray(self.gas,temp.shape)
		gasArray.TDY=temp,rho,massFracs.transpose()

		wf= gasArray.net_production_rates * gasArray.molecular_weights
		return wf[:,:].transpose()



	
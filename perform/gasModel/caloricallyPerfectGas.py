import numpy as np

from perform.constants import REAL_TYPE, R_UNIV, SUTH_TEMP
from perform.gasModel.gasModel import GasModel

# TODO: more options for passing arguments to avoid repeats in called methods

class CaloricallyPerfectGas(GasModel):
	"""
	Container class for all CPG-specific thermo/transport property methods
	"""

	def __init__(self, gasDict):
		super().__init__(gasDict)

		self.enth_ref = gasDict["enth_ref"].astype(REAL_TYPE) 	# reference enthalpy, J/kg
		self.cp       = gasDict["cp"].astype(REAL_TYPE)			# heat capacity at constant pressure, J/(kg-K)
		self.pr       = gasDict["pr"].astype(REAL_TYPE)			# Prandtl number
		self.sc       = gasDict["sc"].astype(REAL_TYPE)			# Schmidt number

		self.mu_ref   = gasDict["mu_ref"].astype(REAL_TYPE)		# reference dynamic viscosity for Sutherland model
		self.temp_ref = gasDict["temp_ref"].astype(REAL_TYPE)	# reference temperature for Sutherland model, K

		assert(self.enth_ref.shape[0] == self.num_species_full)
		assert(self.cp.shape[0] == self.num_species_full)
		assert(self.pr.shape[0] == self.num_species_full)
		assert(self.sc.shape[0] == self.num_species_full)
		assert(self.temp_ref.shape[0] == self.num_species_full)
		assert(self.mu_ref.shape[0] == self.num_species_full)

		self.const_visc_idxs = np.squeeze(np.argwhere(self.temp_ref < 1.0e-7), axis=1)
		self.suth_visc_idxs  = np.squeeze(np.argwhere(self.temp_ref >= 1.0e-7), axis=1)

		self.cp_diffs       = self.cp[self.mass_frac_slice] - self.cp[-1]
		self.enth_ref_diffs = self.enth_ref[self.mass_frac_slice] - self.enth_ref[-1]

	def calc_mix_gas_constant(self, mass_fracs):
		"""
		Compute mixture specific gas constant
		"""

		mass_fracs_in = mass_fracs.copy()
		if (mass_fracs.shape[0] == self.num_species_full):
			mass_fracs_in = mass_fracs_in[self.mass_frac_slice,:]

		r_mix = R_UNIV * ( (1.0 / self.mol_weights[-1]) + np.sum(mass_fracs_in * self.mw_inv_diffs[:,None], axis=0) )
		return r_mix


	def calc_mix_gamma(self, r_mix, cp_mix):
		"""
		Compute mixture ratio of specific heats
		"""

		gamma_mix = cp_mix / (cp_mix - r_mix)
		return gamma_mix


	def calc_mix_enth_ref(self, mass_fracs):
		"""
		Compute mixture reference enthalpy
		"""

		assert(mass_fracs.shape[0] == self.num_species), "Only num_species species must be passed to calc_mix_enth_ref"
		enth_ref_mix = self.enth_ref[-1] + np.sum(mass_fracs * self.enth_ref_diffs[:,None], axis=0)
		return enth_ref_mix


	def calc_mix_cp(self, mass_fracs):
		"""
		Compute mixture specific heat at constant pressure
		"""

		assert(mass_fracs.shape[0] == self.num_species), "Only num_species species must be passed to calc_mix_cp"
		cp_mix = self.cp[-1] + np.sum(mass_fracs * self.cp_diffs[:,None], axis=0)
		return cp_mix

	
	def calc_density(self, sol_prim, r_mix=None):
		"""
		Compute density from ideal gas law
		"""

		# need to calculate mixture gas constant
		if (r_mix is None):
			mass_fracs = self.get_mass_frac_array(sol_prim=sol_prim)
			r_mix = self.calc_mix_gas_constant(mass_fracs)

		# calculate directly from ideal gas
		density = sol_prim[0,:] / (r_mix * sol_prim[2,:])

		return density


	def calc_spec_enth(self, temperature):
		"""
		Compute individual enthalpies for each species
		Returns values for ALL species, NOT num_species species
		"""

		spec_enth = self.cp[:,None] * np.repeat(np.reshape(temperature, (1,-1)), self.num_species_full, axis=0) + self.enth_ref[:,None]

		return spec_enth

	
	def calc_stag_enth(self, velocity, mass_fracs, temperature=None, spec_enth=None):
		"""
		Compute stagnation enthalpy from velocity and species enthalpies
		"""

		# get the species enthalpies if not provided
		if (spec_enth is None):
			assert (temperature is not None), "Must provide temperature if not providing species enthalpies"
			spec_enth = self.calc_spec_enth(temperature)

		# compute all mass fraction fields
		if (mass_fracs.shape[0] == self.num_species):
			mass_fracs = self.calc_all_mass_fracs(mass_fracs, threshold=False)

		stag_enth = np.sum(spec_enth * mass_fracs, axis=0) + 0.5 * np.square(velocity)

		return stag_enth

	
	def calc_species_dynamic_visc(self, temperature):
		"""
		Compute individual dynamic viscosities from Sutherland's law
		Defaults to reference dynamic viscosity if reference temperature is zero
		Returns values for ALL species, NOT num_species species
		"""

		# TODO: theoretically, I think this should account for species-specific Sutherland temperatures

		spec_dyn_visc = np.zeros((self.num_species_full, len(temperature)), dtype=REAL_TYPE)

		# if reference temperature is (close to) zero, constant dynamic viscosity
		if (len(self.const_visc_idxs) > 0):
			spec_dyn_visc[self.const_visc_idxs, :] = self.mu_ref[self.const_visc_idxs, None]

		# otherwise apply Sutherland's law
		if (len(self.suth_visc_idxs) > 0):
			temp_fac = temperature[None,:] / self.temp_ref[self.suth_visc_idxs, None]
			temp_fac = np.power(temp_fac, 3./2.)
			suth_fac = (self.temp_ref[self.suth_visc_idxs, None] + SUTH_TEMP) / (temperature[None, :] + SUTH_TEMP)
			spec_dyn_visc[self.suth_visc_idxs, :] = self.mu_ref[self.suth_visc_idxs, None] * temp_fac * suth_fac

		return spec_dyn_visc

	
	def calc_mix_dynamic_visc(self, spec_dyn_visc=None, temperature=None, mole_fracs=None, mass_fracs=None, mw_mix=None):
		"""
		Compute mixture dynamic viscosity from Wilkes mixing law
		"""

		if (spec_dyn_visc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			spec_dyn_visc = self.calc_species_dynamic_visc(temperature)

		if (self.num_species_full == 1):

			mix_dyn_visc = np.squeeze(spec_dyn_visc)

		else:

			if (mole_fracs is None):
				assert (mass_fracs is not None), "Must provide mass fractions if not providing mole fractions"
				mole_fracs = self.calc_all_mole_fracs(mass_fracs, mw_mix)

			phi = np.zeros((self.num_species_full, spec_dyn_visc.shape[1]), dtype=REAL_TYPE)
			for spec_idx in range(self.num_species_full):

				muFac = np.sqrt(spec_dyn_visc[[spec_idx],:] / spec_dyn_visc)
				phi[spec_idx, :] = np.sum(mole_fracs * np.square(1.0 + muFac * self.mix_mass_matrix[[spec_idx],:].T) * self.mix_inv_mass_matrix[[spec_idx],:].T, axis=0)

			mix_dyn_visc = np.sum( mole_fracs * spec_dyn_visc / phi, axis=0)

		return mix_dyn_visc

	
	def calc_species_therm_cond(self, spec_dyn_visc=None, temperature=None):
		"""
		Compute species thermal conductivities
		Returns values for ALL species, NOT num_species species
		"""

		if (spec_dyn_visc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			spec_dyn_visc = self.calc_species_dynamic_visc(temperature)

		spec_therm_cond = spec_dyn_visc * self.cp[:, None] / self.pr[:, None]

		return spec_therm_cond

	
	def calc_mix_thermal_cond(self, spec_therm_cond=None, spec_dyn_visc=None, temperature=None, mole_fracs=None, mass_fracs=None, mw_mix=None):
		"""
		Compute mixture thermal conductivity
		"""

		if (spec_therm_cond is None):
			assert ((spec_dyn_visc is not None) or (temperature is not None)), \
					"Must provide species dynamic viscosity or temperature if not providing species thermal conductivity"
			spec_therm_cond = self.calc_species_therm_cond(spec_dyn_visc=spec_dyn_visc, temperature=temperature)

		if (self.num_species_full == 1):

			mix_therm_cond = np.squeeze(spec_therm_cond)

		else:

			if (mole_fracs is None):
				assert (mass_fracs is not None), "Must provide mass fractions if not providing mole fractions"
				mole_fracs = self.calc_all_mole_fracs(mass_fracs, mix_mol_weight=mw_mix)

			mix_therm_cond = 0.5 * ( np.sum(mole_fracs * spec_therm_cond, axis=0) + 1.0 / np.sum(mole_fracs / spec_therm_cond, axis=0) )

		return mix_therm_cond

	
	def calc_species_mass_diff_coeff(self, density, spec_dyn_visc=None, temperature=None):
		"""
		Compute mass diffusivity coefficient of species into mixture
		Returns values for ALL species, NOT num_species species
		"""

		if (spec_dyn_visc is None):
			assert (temperature is not None), "Must provide temperature if not providing species dynamic viscosities"
			spec_dyn_visc = self.calc_species_dynamic_visc(temperature)
		
		spec_mass_diff = spec_dyn_visc / (self.sc[:, None] * density[None, :])

		return spec_mass_diff

	
	def calc_sound_speed(self, temperature, r_mix=None, gamma_mix=None, mass_fracs=None, cp_mix=None):
		"""
		Compute sound speed
		"""

		# calculate mixture gas constant if not provided
		mass_fracs_set = False
		if (r_mix is None):
			assert (mass_fracs is not None), "Must provide mass fractions to calculate mixture gas constant..."
			mass_fracs = self.get_mass_frac_array(mass_fracs=mass_fracs)
			mass_fracs_set = True
			r_mix = self.calc_mix_gas_constant(mass_fracs)
		else:
			r_mix = np.squeeze(r_mix)
			
		# calculate ratio of specific heats if not provided
		if (gamma_mix is None):
			if (cp_mix is None):
				assert (mass_fracs is not None), "Must provide mass fractions to calculate mixture cp..."
				if (not mass_fracs_set): 
					mass_fracs = self.get_mass_frac_array(mass_fracs=mass_fracs)
				cp_mix = self.calc_mix_cp(mass_fracs)
			else:
				cp_mix = np.squeeze(cp_mix)

			gamma_mix = self.calc_mix_gamma(r_mix, cp_mix)
		else:
			gamma_mix = np.squeeze(gamma_mix)

		sound_speed = np.sqrt(gamma_mix * r_mix * temperature)

		return sound_speed

	
	def calc_dens_derivs(self, density, 
								wrt_press=False, pressure=None,
								wrt_temp=False, temperature=None,
								wrt_spec=False, mix_mol_weight=None, mass_fracs=None):

		"""
		Compute derivatives of density with respect to pressure, temperature, or species mass fraction
		For species derivatives, returns num_species derivatives
		"""

		assert any([wrt_press, wrt_temp, wrt_spec]), "Must compute at least one density derivative..."

		derivs = tuple()
		if (wrt_press):
			assert (pressure is not None), "Must provide pressure for pressure derivative..."
			d_dens_d_press = density / pressure
			derivs = derivs + (d_dens_d_press,)

		if (wrt_temp):
			assert (temperature is not None), "Must provide temperature for temperature derivative..."
			d_dens_d_temp = -density / temperature
			derivs = derivs + (d_dens_d_temp,)

		if (wrt_spec):
			# calculate mixture molecular weight
			if (mix_mol_weight is None):
				assert (mass_fracs is not None), "Must provide mass fractions to calculate mixture mol weight..."
				mix_mol_weight = self.calc_mix_mol_weight(mass_fracs)

			d_dens_d_mass_frac = np.zeros((self.num_species, density.shape[0]), dtype=REAL_TYPE)
			for spec_idx in range(self.num_species):
				d_dens_d_mass_frac[spec_idx, :] = density * mix_mol_weight * (1.0 / self.mol_weights[-1] - 1.0 / self.mol_weights[spec_idx])
			derivs = derivs + (d_dens_d_mass_frac,)

		return derivs

	
	def calc_stag_enth_derivs(self, wrt_press=False,
							  wrt_temp=False, mass_fracs=None,
							  wrt_vel=False, velocity=None,
							  wrt_spec=False, spec_enth=None, temperature=None):

		"""
		Compute derivatives of stagnation enthalpy with respect to pressure, temperature, velocity, or species mass fraction
		For species derivatives, returns num_species derivatives
		"""

		assert any([wrt_press, wrt_temp, wrt_vel, wrt_spec]), "Must compute at least one density derivative..."

		derivs = tuple()
		if (wrt_press):
			d_stag_enth_d_press = 0.0
			derivs = derivs + (d_stag_enth_d_press,)
		
		if (wrt_temp):
			assert (mass_fracs is not None), "Must provide mass fractions for temperature derivative..."

			if (mass_fracs.shape[0] != self.num_species):
				mass_fracs = self.get_mass_frac_array(mass_fracs=mass_fracs)
			d_stag_enth_d_temp = self.calc_mix_cp(mass_fracs)
			derivs = derivs + (d_stag_enth_d_temp,)

		if (wrt_vel):
			assert (velocity is not None), "Must provide velocity for velocity derivative..."
			d_stag_enth_d_vel = velocity.copy()
			derivs = derivs + (d_stag_enth_d_vel,)

		if (wrt_spec):
			if (spec_enth is None):
				assert (temperature is not None), "Must provide temperature if not providing species enthalpies..."
				spec_enth = self.calc_spec_enth(temperature)
			
			d_stag_enth_d_mass_frac = np.zeros((self.num_species, spec_enth.shape[1]), dtype=REAL_TYPE)
			if (self.num_species_full == 1):
				d_stag_enth_d_mass_frac[0,:] = spec_enth[0,:]
			else:
				for spec_idx in range(self.num_species):
					d_stag_enth_d_mass_frac[spec_idx,:] = spec_enth[spec_idx,:] - spec_enth[-1,:]

			derivs = derivs + (d_stag_enth_d_mass_frac,)

		return derivs
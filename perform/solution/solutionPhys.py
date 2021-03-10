import numpy as np

from perform.constants import REAL_TYPE, HUGE_NUM


class SolutionPhys:
	"""
	Base class for physical solution (opposed to ROM solution)
	"""

	def __init__(self, gas, num_cells, sol_prim_in=None, sol_cons_in=None):
		
		self.gas_model = gas

		self.num_cells = num_cells

		# primitive and conservative state
		self.sol_prim = np.zeros((self.gas_model.num_eqs, num_cells), dtype=REAL_TYPE)		# solution in primitive variables
		self.sol_cons = np.zeros((self.gas_model.num_eqs, num_cells), dtype=REAL_TYPE)		# solution in conservative variables
		
		# chemical properties
		self.mw_mix    = np.zeros(num_cells, dtype=REAL_TYPE)								# mixture molecular weight
		self.r_mix     = np.zeros(num_cells, dtype=REAL_TYPE)								# mixture specific gas constant
		self.gamma_mix = np.zeros(num_cells, dtype=REAL_TYPE)								# mixture ratio of specific heats
		
		# thermodynamic properties
		self.enth_ref_mix = np.zeros(num_cells, dtype=REAL_TYPE)								# mixture reference enthalpy
		self.cp_mix       = np.zeros(num_cells, dtype=REAL_TYPE)								# mixture specific heat at constant pressure
		self.h0           = np.zeros(num_cells, dtype=REAL_TYPE) 								# stagnation enthalpy
		self.hi           = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)# species enthalpies
		self.c            = np.zeros(num_cells, dtype=REAL_TYPE) 								# sound speed

		# transport properties
		self.dyn_visc_mix   = np.zeros(num_cells, dtype=REAL_TYPE)									# mixture dynamic viscosity
		self.therm_cond_mix = np.zeros(num_cells, dtype=REAL_TYPE) 									# mixture thermal conductivity
		self.mass_diff_mix  = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE) 	# mass diffusivity coefficient (into mixture)

		# derivatives of density and enthalpy
		self.d_rho_d_press      = np.zeros(num_cells, dtype=REAL_TYPE)
		self.d_rho_d_temp       = np.zeros(num_cells, dtype=REAL_TYPE)
		self.d_rho_d_mass_frac  = np.zeros((self.gas_model.num_species, num_cells), dtype=REAL_TYPE)
		self.d_enth_d_press     = np.zeros(num_cells, dtype=REAL_TYPE)
		self.d_enth_d_temp      = np.zeros(num_cells, dtype=REAL_TYPE)
		self.d_enth_d_mass_frac = np.zeros((self.gas_model.num_species, num_cells), dtype=REAL_TYPE)	

		# reaction rate-of-progress variables
		# TODO: generalize to >1 reaction, reverse reactions
		self.wf = np.zeros((1, num_cells), dtype=REAL_TYPE)

		# set initial condition
		if (sol_prim_in is not None):
			assert(sol_prim_in.shape == (self.gas_model.num_eqs, num_cells))
			self.sol_prim = sol_prim_in.copy()
			self.update_state(from_cons=False)
		elif (sol_cons_in is not None):
			assert(sol_cons_in.shape == (self.gas_model.num_eqs, num_cells))
			self.sol_cons = sol_cons_in.copy()
			self.update_state(from_cons=True)
		else:
			raise ValueError("Must provide either sol_prim_in or sol_cons_in to solutionPhys")

	def update_state(self, from_cons=True):
		"""
		Update state and some mixture gas properties
		"""

		if from_cons:
			self.calc_state_from_cons(calc_r=True, calc_cp=True, calc_gamma=True)
		else:
			self.calc_state_from_prim(calc_r=True, calc_cp=True, calc_gamma=True)


	def calc_state_from_cons(self, calc_r=False, calc_cp=False, calc_gamma=False):
		"""
		Compute primitive state from conservative state
		"""

		self.sol_prim[3:,:] = self.sol_cons[3:,:] / self.sol_cons[[0],:]
		mass_fracs = self.gas_model.get_mass_frac_array(sol_prim=self.sol_prim)

		# threshold
		# TODO: is this valid? It shouldn't violate mass conservation since density stays the same
		mass_fracs = self.gas_model.calc_all_mass_fracs(mass_fracs, threshold=True)
		if (self.gas_model.num_species_full > 1):
			mass_fracs = mass_fracs[:-1,:]
		self.sol_prim[3:,:] = mass_fracs
		self.sol_cons[3:,:] = self.sol_prim[3:,:] * self.sol_cons[[0],:]

		# update thermo properties
		self.enth_ref_mix = self.gas_model.calc_mix_enth_ref(mass_fracs)
		if calc_gamma:
			calc_r  = True
			calc_cp = True
		if calc_r:     self.r_mix     = self.gas_model.calc_mix_gas_constant(mass_fracs)
		if calc_cp:    self.cp_mix    = self.gas_model.calc_mix_cp(mass_fracs)
		if calc_gamma: self.gamma_mix = self.gas_model.calc_mix_gamma(self.r_mix,self.cp_mix)

		# update primitive state
		# TODO: gas_model references
		self.sol_prim[1,:] = self.sol_cons[1,:] / self.sol_cons[0,:]
		self.sol_prim[2,:] = (self.sol_cons[2,:] / self.sol_cons[0,:] - np.square(self.sol_prim[1,:]) / 2.0 - 
							 self.enth_ref_mix) / (self.cp_mix - self.r_mix) 
		self.sol_prim[0,:] = self.sol_cons[0,:] * self.r_mix * self.sol_prim[2,:]


	def calc_state_from_prim(self, calc_r=False, calc_cp=False, calc_gamma=False):
		"""
		Compute state from primitive state
		"""

		mass_fracs = self.gas_model.get_mass_frac_array(sol_prim=self.sol_prim)
		# threshold
		mass_fracs = self.gas_model.calc_all_mass_fracs(mass_fracs, threshold=True)
		if (self.gas_model.num_species_full > 1):
			mass_fracs = mass_fracs[:-1,:]
		self.sol_prim[3:,:] = mass_fracs

		# update thermo properties
		self.enth_ref_mix = self.gas_model.calc_mix_enth_ref(mass_fracs)
		if calc_gamma:
			calc_r  = True
			calc_cp = True
		if calc_r:       self.r_mix       = self.gas_model.calc_mix_gas_constant(mass_fracs)
		if calc_cp:      self.cp_mix      = self.gas_model.calc_mix_cp(mass_fracs)
		if calc_gamma:   self.gamma_mix   = self.gas_model.calc_mix_gamma(self.r_mix,self.cp_mix)

		# update conservative variables
		# TODO: gas_model references
		self.sol_cons[0,:]  = self.sol_prim[0,:] / (self.r_mix * self.sol_prim[2,:]) 
		self.sol_cons[1,:]  = self.sol_cons[0,:] * self.sol_prim[1,:]				
		self.sol_cons[2,:]  = self.sol_cons[0,:] * ( self.enth_ref_mix + self.cp_mix * self.sol_prim[2,:] + 
												 np.power(self.sol_prim[1,:], 2.0) / 2.0 ) - self.sol_prim[0,:]
		self.sol_cons[3:,:] = self.sol_cons[[0],:] * self.sol_prim[3:,:]
 

	def calc_state_from_rho_h0(self):
		"""
		Adjust pressure and temperature iteratively to agree with a fixed density and stagnation enthalpy
		Used to compute a physically-meaningful Roe average state from the Roe average enthalpy and density
		"""

		# TODO: some of this changes for TPG

		rho_fixed = np.squeeze(self.sol_cons[0,:])
		h0_fixed  = np.squeeze(self.h0)

		d_press = HUGE_NUM * np.ones(self.num_cells, dtype=REAL_TYPE)
		d_temp  = HUGE_NUM * np.ones(self.num_cells, dtype=REAL_TYPE)

		iter_count = 0
		ones_vec = np.ones(self.num_cells, dtype=REAL_TYPE)
		while ( (np.any( np.absolute(d_press / self.sol_prim[0,:]) > 0.01 ) or \
				 np.any( np.absolute(d_temp / self.sol_prim[2,:]) > 0.01)) and \
				 (iter_count < 20) ):

			# compute density and stagnation enthalpy from current state
			dens_curr = self.gas_model.calc_density(self.sol_prim)
			h0_curr   = self.gas_model.calc_stag_enth(self.sol_prim)

			# compute difference between current and fixed density/stagnation enthalpy
			d_dens      = rho_fixed - dens_curr 
			d_stag_enth = h0_fixed - h0_curr

			# compute derivatives of density and stagnation enthalpy with respect to pressure and temperature
			d_dens_d_press, d_dens_d_temp = \
				self.gas_model.calc_dens_derivs(dens_curr, 
												wrt_press=True, pressure=self.sol_prim[0,:], 
												wrt_temp=True, temperature=self.sol_prim[2,:])

			d_stag_enth_d_press, d_stag_enth_d_temp = \
				self.gas_model.calc_stag_enth_derivs(wrt_press=True, wrt_temp=True, mass_fracs=self.sol_prim[3:,:])

			# compute change in temperature and pressure 
			d_factor = 1.0 / (d_dens_d_press * d_stag_enth_d_temp - d_dens_d_temp * d_stag_enth_d_press)
			d_press  = d_factor * (d_dens * d_stag_enth_d_temp - d_stag_enth * d_dens_d_temp)
			d_temp   = d_factor * (-d_dens * d_stag_enth_d_press + d_stag_enth * d_dens_d_press)

			# threshold change in temperature and pressure 
			d_press = np.copysign(ones_vec, d_press) * np.minimum(np.absolute(d_press), self.sol_prim[0,:] * 0.1)
			d_temp 	= np.copysign(ones_vec, d_temp) * np.minimum(np.absolute(d_temp), self.sol_prim[2,:] * 0.1)

			# update temperature and pressure
			self.sol_prim[0,:] += d_press
			self.sol_prim[2,:] += d_temp

			iter_count += 1

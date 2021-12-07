import numpy as np

from perform.constants import REAL_TYPE, HUGE_NUM


class SolutionPhys:
    """Base class for full-dimensional physical solution.

    This class provides the attributes and member functions required for describing a discrete system solution,
    including transport properties, thermodynamic properties, and derivative thereof. This is general to any required
    solution representation and is not restricted to cell-centered values. For example, this may be used to represent
    face reconstruction states.

    Args:
        gas: GasModel associated with the SolutionDomain with which this SolutionPhys is associated.
        num_cells: Number of finite volume cells represented by this SolutionPhys.
        sol_prim_in: NumPy array of the primitive state profiles that this SolutionPhys represents.
        sol_cons_in: NumPy array of the conservative state profiles that this SolutionPhys represents.

    Attributes:
        gas_model: GasModel associated with the SolutionDomain with which this SolutionPhys is associated.
        num_cells: Number of finite volume cells represented by this SolutionPhys.
        sol_prim: NumPy array of the primitive state profiles that this SolutionPhys represents.
        sol_cons: NumPy array of the conservative state profiles that this SolutionPhys represents.
        mass_fracs_full: NumPy array of all num_species_full mass fraction profiles.
        mw_mix: NumPy array of the mixture molecular weight profile, in g/mol.
        r_mix: NumPy array of the mixture specific gas constant profile, in J/kg-K.
        gamma_mix: NumPy array of the mixture ratio of specific heats profile.
        enth_ref_mix: NumPy array of the mixture reference enthalpy profile, in J/kg.
        cp_mix: NumPy array of the mixture specific heat capacity at constant pressure profile, in J/kg-K.
        h0: NumPy array of the stangation enthalpy profile, in J/kg.
        hi: NumPy array of the species enthalpy profiles for all num_species_full chemical species, in J/kg.
        c: NumPy array of the sound speed profile, in m/s.
        dyn_visc_mix: NumPy array of the mixture dynamic viscosity profile, in N-s/m^2.
        therm_cond_mix: NumPy array of the mixture thermal conductivity, in W/m-K.
        mass_diff_mix:
            NumPy array of the mass diffusivity profiles for all num_species_full chemical species, in m^2/s.
        d_rho_d_press: NumPy array of derivative of density w/r/t pressure profile.
        d_rho_d_temp: NumPy array of derivative of density w/r/t temperature profile.
        d_rho_d_mass_frac: NumPy array of derivative of density w/r/t the first num_species mass fraction profiles.
        d_enth_d_press: NumPy array of derivative of stagnation enthalpy w/r/t pressure profile.
        d_enth_d_temp: NumPy array of derivative of stagnation enthalpy w/r/t temperature profile.
        d_enth_d_mass_frac:
            NumPy array of derivative of stagnation enthalpy w/r/t the first num_species mass fraction profiles.
    """

    def __init__(self, gas, num_cells, sol_prim_in=None, sol_cons_in=None, time_order=1):

        self.gas_model = gas

        self.num_cells = num_cells

        # Primitive and conservative state
        self.sol_prim = np.zeros((self.gas_model.num_eqs, num_cells), dtype=REAL_TYPE)
        self.sol_cons = np.zeros((self.gas_model.num_eqs, num_cells), dtype=REAL_TYPE)

        # Chemical composition
        self.mass_fracs_full = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)
        self.mole_fracs_full = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)
        self.mw_mix = np.zeros(num_cells, dtype=REAL_TYPE)

        # Thermodynamic properties
        self.enth_ref_mix = np.zeros(num_cells, dtype=REAL_TYPE)
        self.r_mix = np.zeros(num_cells, dtype=REAL_TYPE)
        self.gamma_mix = np.zeros(num_cells, dtype=REAL_TYPE)
        self.cp_mix = np.zeros(num_cells, dtype=REAL_TYPE)
        self.h0 = np.zeros(num_cells, dtype=REAL_TYPE)
        self.hi = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)
        self.c = np.zeros(num_cells, dtype=REAL_TYPE)

        # Transport properties
        self.spec_dyn_visc = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)
        self.dyn_visc_mix = np.zeros(num_cells, dtype=REAL_TYPE)
        self.spec_therm_cond = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)
        self.therm_cond_mix = np.zeros(num_cells, dtype=REAL_TYPE)
        self.mass_diff_mix = np.zeros((self.gas_model.num_species_full, num_cells), dtype=REAL_TYPE)

        # Derivatives of density and enthalpy
        self.d_rho_d_press = np.zeros(num_cells, dtype=REAL_TYPE)
        self.d_rho_d_temp = np.zeros(num_cells, dtype=REAL_TYPE)
        self.d_rho_d_mass_frac = np.zeros((self.gas_model.num_species, num_cells), dtype=REAL_TYPE)
        self.d_enth_d_press = np.zeros(num_cells, dtype=REAL_TYPE)
        self.d_enth_d_temp = np.zeros(num_cells, dtype=REAL_TYPE)
        self.d_enth_d_mass_frac = np.zeros((self.gas_model.num_species, num_cells), dtype=REAL_TYPE)

        # Compute complete initial state history and set initial condition
        self.sol_hist_cons = [None] * (time_order + 1)
        self.sol_hist_prim = [None] * (time_order + 1)
        if sol_prim_in is not None:
            # account for inputs without time dimension
            if sol_prim_in.ndim == 2:
                sol_prim_in = np.expand_dims(sol_prim_in, -1)
            init_snaps = sol_prim_in.shape[-1]
            assert init_snaps <= time_order, "CRW: Need to fix initialization loop"

            # set "history" component of sol_hist_*
            for init_idx in range(0, init_snaps):
                # NOTE: have to reverse order of snapshots
                self.sol_prim = sol_prim_in[:, :, init_idx].copy()
                assert self.sol_prim.shape == (self.gas_model.num_eqs, num_cells)
                self.update_state(from_prim=True)
                snap_idx = init_snaps - init_idx
                self.sol_hist_prim[snap_idx] = self.sol_prim.copy()
                self.sol_hist_cons[snap_idx] = self.sol_cons.copy()

        # mirror of above, for conservative initial state
        elif sol_cons_in is not None:
            if sol_cons_in.ndim == 2:
                sol_cons_in = np.expand_dims(sol_cons_in, -1)
            init_snaps = sol_cons_in.shape[-1]
            assert init_snaps <= time_order, "CRW: Need to fix initialization loop"

            for init_idx in range(0, init_snaps):
                # NOTE: have to reverse order of snapshots
                self.sol_cons = sol_cons_in[:, :, init_idx].copy()
                assert self.sol_cons.shape == (self.gas_model.num_eqs, num_cells)
                self.update_state(from_prim=False)
                snap_idx = init_snaps - init_idx
                self.sol_hist_prim[snap_idx] = self.sol_prim.copy()
                self.sol_hist_cons[snap_idx] = self.sol_cons.copy()

        else:
            raise ValueError("Must provide either sol_prim_in or sol_cons_in to SolutionPhys")

        # initial guess for next step
        self.sol_hist_prim[0] = self.sol_hist_prim[1].copy()
        self.sol_hist_cons[0] = self.sol_hist_cons[1].copy()

        # fill in any remaining snapshots in history
        for snap_idx in range(init_snaps + 1, time_order + 1):
            self.sol_hist_prim[snap_idx] = self.sol_hist_prim[init_snaps].copy()
            self.sol_hist_cons[snap_idx] = self.sol_hist_cons[init_snaps].copy()

    def update_state(self, from_prim):
        """Utility function to complete state from primitive or conservative solution.

        Calculates chemical composition, thermodynamic properties, transport properties,
        and primitive/conservative solution from the conservative/primitive solution.

        Args:
            from_prim:
                If False, update primitive state from conservative state.
                If True, update conservative state from primitive state.
        """

        # Get mass fractions
        if not from_prim:
            self.sol_prim[3:, :] = self.sol_cons[3:, :] / self.sol_cons[[0], :]

        # Update chemical composition and thermo properties
        self.update_chemical_composition()
        self.update_thermo_properties()

        # Update primitive/conservative state
        if from_prim:
            self.update_cons_from_prim()
        else:
            self.update_prim_from_cons()

        # Finally, transport properties
        self.update_transport_properties()

    def update_prim_from_cons(self):
        """Update primitive solution from the conservative solution.

        Assumes that thermodynamic properties have already been updated.
        """

        self.sol_prim[1, :] = self.sol_cons[1, :] / self.sol_cons[0, :]
        self.sol_prim[0, :], self.sol_prim[2, :] = self.gas_model.calc_press_temp_from_cons(
            self.sol_cons[0, :],
            self.sol_cons[2, :],
            velocity=self.sol_prim[1, :],
            cp_mix=self.cp_mix,
            r_mix=self.r_mix,
            enth_ref_mix=self.enth_ref_mix,
        )

    def calc_prim_from_cons(self, sol_cons):
        """Calculate and return primitive solution from given conservative solution.

        Computes all required thermodynamic quantities internally, and does not modify the calling SolutionPhys.
        """

        sol_prim = np.zeros(sol_cons.shape, dtype=sol_cons.dtype)

        # species mass fraction and velocity
        sol_prim[3:, :] = sol_cons[3:, :] / sol_cons[[0], :]
        sol_prim[1, :] = sol_cons[1, :] / sol_cons[0, :]

        # thermodynamic quantities
        enth_ref_mix = self.gas_model.calc_mix_enth_ref(sol_prim[3:, :])
        r_mix = self.gas_model.calc_mix_gas_constant(sol_prim[3:, :])
        cp_mix = self.gas_model.calc_mix_cp(sol_prim[3:, :])

        # pressure and temperature
        sol_prim[0, :], sol_prim[2, :] = self.gas_model.calc_press_temp_from_cons(
            sol_cons[0, :],
            sol_cons[2, :],
            velocity=sol_prim[1, :],
            cp_mix=cp_mix,
            r_mix=r_mix,
            enth_ref_mix=enth_ref_mix,
        )

        return sol_prim

    def update_cons_from_prim(self):
        """Update conservative solution from the primitive solution.

        Assumes that thermodynamic properties have already been updated
        """

        self.sol_cons[0, :] = self.sol_prim[0, :] / (self.r_mix * self.sol_prim[2, :])
        self.sol_cons[1, :] = self.sol_cons[0, :] * self.sol_prim[1, :]
        self.sol_cons[2, :] = self.sol_cons[0, :] * self.h0 - self.sol_prim[0, :]
        self.sol_cons[3:, :] = self.sol_cons[[0], :] * self.sol_prim[3:, :]

    def calc_cons_from_prim(self, sol_prim):
        """Calculate and return conservative solution from given primitive solution.

        Computes all required thermodynamic quantities internally, and does not modify the calling SolutionPhys.
        """

        sol_cons = np.zeros(sol_prim.shape, dtype=sol_prim.dtype)

        # chemical composition
        mass_fracs = self.gas_model.get_mass_frac_array(sol_prim_in=sol_prim)
        mass_fracs_full = self.gas_model.calc_all_mass_fracs(mass_fracs, threshold=True)

        # thermodynamic quantities
        hi = self.gas_model.calc_spec_enth(sol_prim[2, :])
        h0 = self.gas_model.calc_stag_enth(sol_prim[1, :], mass_fracs_full, spec_enth=hi)
        r_mix = self.gas_model.calc_mix_gas_constant(sol_prim[3:, :])

        # conservative state
        sol_cons[0, :] = sol_prim[0, :] / (r_mix * sol_prim[2, :])
        sol_cons[1, :] = sol_cons[0, :] * sol_prim[1, :]
        sol_cons[2, :] = sol_cons[0, :] * h0 - sol_prim[0, :]
        sol_cons[3:, :] = sol_cons[[0], :] * sol_prim[3:, :]

        return sol_cons

    def update_chemical_composition(self):
        """Update complete chemical composition from num_species species mass fractions.

        Calculates num_species_full-th species mass fraction, mixture molecular weights, and mole fractions.
        """

        # Compute all mass fraction fields
        mass_fracs = self.gas_model.get_mass_frac_array(sol_prim_in=self.sol_prim)
        mass_fracs = self.gas_model.calc_all_mass_fracs(mass_fracs, threshold=True)
        self.mass_fracs_full[:, :] = mass_fracs.copy()
        if self.gas_model.num_species_full > 1:
            mass_fracs = mass_fracs[:-1, :]
        self.sol_prim[3:, :] = mass_fracs

        self.mw_mix[:] = self.gas_model.calc_mix_mol_weight(self.mass_fracs_full)
        self.mole_fracs_full[:, :] = self.gas_model.calc_all_mole_fracs(
            self.mass_fracs_full, mix_mol_weight=self.mw_mix
        )

    def update_thermo_properties(self):
        """Update thermodynamic properties.

        Calculates the mixture specific gas constant, mixture specific heat capacity at constant pressure,
        mixture ratio of specific heats, sound speed, species enthalpies, and stagnation enthalpy.
        """

        self.enth_ref_mix[:] = self.gas_model.calc_mix_enth_ref(self.sol_prim[3:, :])
        self.r_mix[:] = self.gas_model.calc_mix_gas_constant(self.sol_prim[3:, :])
        self.cp_mix[:] = self.gas_model.calc_mix_cp(self.sol_prim[3:, :])
        self.gamma_mix[:] = self.gas_model.calc_mix_gamma(r_mix=self.r_mix, cp_mix=self.cp_mix)

        self.c[:] = self.gas_model.calc_sound_speed(self.sol_prim[2, :], r_mix=self.r_mix, gamma_mix=self.gamma_mix)
        self.hi[:, :] = self.gas_model.calc_spec_enth(self.sol_prim[2, :])
        self.h0[:] = self.gas_model.calc_stag_enth(self.sol_prim[1, :], self.mass_fracs_full, spec_enth=self.hi)

    def update_transport_properties(self):
        """Update transport properties.

        Calculates species and mixture dynamic viscosity, species and mixture thermal conductivity, and the
        mixture mass diffusivity.
        """

        self.spec_dyn_visc[:, :] = self.gas_model.calc_species_dynamic_visc(self.sol_prim[2, :])
        self.dyn_visc_mix[:] = self.gas_model.calc_mix_dynamic_visc(
            spec_dyn_visc=self.spec_dyn_visc, mole_fracs=self.mole_fracs_full, mw_mix=self.mw_mix
        )
        self.spec_therm_cond[:, :] = self.gas_model.calc_species_therm_cond(spec_dyn_visc=self.spec_dyn_visc)
        self.therm_cond_mix[:] = self.gas_model.calc_mix_thermal_cond(
            spec_therm_cond=self.spec_therm_cond,
            spec_dyn_visc=self.spec_dyn_visc,
            mole_fracs=self.mole_fracs_full,
            mw_mix=self.mw_mix,
        )
        self.mass_diff_mix[:, :] = self.gas_model.calc_species_mass_diff_coeff(
            self.sol_cons[0, :], spec_dyn_visc=self.spec_dyn_visc
        )

    def update_density_enthalpy_derivs(self):
        """Updates density and enthalpy derivates w/r/t pressure, temperature, and species mass fraction.

        This function is not called as part of calc_state_from_prim()/cons() as it is only required from implicit
        time integration, and is also not required for all states. These derivatives can be a bit expensive to
        calculate, so this function should only be called where absolutely necessary.
        """

        gas = self.gas_model

        # Density derivatives
        self.d_rho_d_press[:], self.d_rho_d_temp[:], self.d_rho_d_mass_frac[:, :] = gas.calc_dens_derivs(
            self.sol_cons[0, :],
            wrt_press=True,
            pressure=self.sol_prim[0, :],
            wrt_temp=True,
            temperature=self.sol_prim[2, :],
            wrt_spec=True,
            mix_mol_weight=self.mw_mix[:],
        )

        # Stagnation enthalpy derivatives
        self.d_enth_d_press[:], self.d_enth_d_temp[:], self.d_enth_d_mass_frac[:, :] = gas.calc_stag_enth_derivs(
            wrt_press=True,
            wrt_temp=True,
            mass_fracs=self.sol_prim[3:, :],
            wrt_spec=True,
            spec_enth=self.hi[:, :],
        )

    def calc_state_from_rho_h0(self):
        """Iteratively solve for pressure and temperature given fixed density and stagnation enthalpy.

        Used to compute a physically-meaningful Roe average state from the Roe average enthalpy and density.
        The simple Roe average of all primitive fields, stagnation enthalpy, and density will
        result in an inconsistent state description otherwise.
        """

        rho_fixed = np.squeeze(self.sol_cons[0, :])
        h0_fixed = np.squeeze(self.h0)

        d_press = HUGE_NUM * np.ones(self.num_cells, dtype=REAL_TYPE)
        d_temp = HUGE_NUM * np.ones(self.num_cells, dtype=REAL_TYPE)

        iter_count = 0
        ones_vec = np.ones(self.num_cells, dtype=REAL_TYPE)
        while (
            np.any(np.absolute(d_press / self.sol_prim[0, :]) > 0.01)
            or np.any(np.absolute(d_temp / self.sol_prim[2, :]) > 0.01)
        ) and (iter_count < 20):

            # Compute density and stagnation enthalpy from current state
            dens_curr = self.gas_model.calc_density(self.sol_prim)

            hi_curr = self.gas_model.calc_spec_enth(self.sol_prim[2, :])
            h0_curr = self.gas_model.calc_stag_enth(self.sol_prim[1, :], self.mass_fracs_full, spec_enth=hi_curr)

            # Compute difference between current and fixed density/stagnation enthalpy
            d_dens = rho_fixed - dens_curr
            d_stag_enth = h0_fixed - h0_curr

            # Compute derivatives of density and stagnation enthalpy with respect to pressure and temperature
            d_dens_d_press, d_dens_d_temp = self.gas_model.calc_dens_derivs(
                dens_curr, wrt_press=True, pressure=self.sol_prim[0, :], wrt_temp=True, temperature=self.sol_prim[2, :]
            )

            d_stag_enth_d_press, d_stag_enth_d_temp = self.gas_model.calc_stag_enth_derivs(
                wrt_press=True, wrt_temp=True, mass_fracs=self.mass_fracs_full
            )

            # Compute change in temperature and pressure
            d_factor = 1.0 / (d_dens_d_press * d_stag_enth_d_temp - d_dens_d_temp * d_stag_enth_d_press)
            d_press = d_factor * (d_dens * d_stag_enth_d_temp - d_stag_enth * d_dens_d_temp)
            d_temp = d_factor * (-d_dens * d_stag_enth_d_press + d_stag_enth * d_dens_d_press)

            # Threshold change in temperature and pressure
            d_press = np.copysign(ones_vec, d_press) * np.minimum(np.absolute(d_press), self.sol_prim[0, :] * 0.1)
            d_temp = np.copysign(ones_vec, d_temp) * np.minimum(np.absolute(d_temp), self.sol_prim[2, :] * 0.1)

            # Update temperature and pressure
            self.sol_prim[0, :] += d_press
            self.sol_prim[2, :] += d_temp

            iter_count += 1

        self.update_state(from_prim=True)

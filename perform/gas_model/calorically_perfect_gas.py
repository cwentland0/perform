import numpy as np

from perform.constants import REAL_TYPE, R_UNIV, SUTH_TEMP
from perform.gas_model.gas_model import GasModel

# TODO: more options for passing arguments to avoid repeats in called methods


class CaloricallyPerfectGas(GasModel):
    """Class implementing all CPG-specific thermo/transport property methods.

    This class provides public methods for calculating various thermodynamic and transport properties,
    as well as some extra utility functions such as density and enthalpy derivatives. Please refer to the
    solver theory documentation for more details on how each quantity is calculated.

    Args:
        chem_dict: Dictionary of input parameters read from chemistry file.

    Attributes:
        enth_ref: NumPy array of reference enthalpies for all num_species_full species, in J/kg.
        cp: NumPy array of specific heat at constant pressure for all num_species_full species, in J/kg-K.
        pr: NumPy array of Prandtl numbers for all num_species_full species.
        sc: NumPy array of Schmidt numbers for all num_species_full species.
        mu_ref:
            NumPy array of reference dynamic viscosities for all num_species_full species for computing
            dynamic viscosity via Sutherland's law, in N-s/m^2.
        temp_ref:
            NumPy array of reference temperatures for all num_species_full species for computing
            dynamic viscosity via Sutherland's law, in Kelvin.
        const_visc_idxs: NumPy array of indices for slicing chemical species with constant dynamic viscosity.
        suth_visc_idxs:
            NumPy array of indices for slicing chemical species with temperature-dependent dynamic viscosity.
        cp_diffs:
            Differences between first num_species specific heat and last species' specific heat.
            Saves cost in computing mixture specific heat, in J/kg-K.
        enth_ref_diffs:
            Differences between first num_species reference enthalpies and last species' reference enthalpy.
            Saves cost in computing mixture reference enthalpy, in J/kg.
    """

    def __init__(self, chem_dict):
        super().__init__(chem_dict)

        self.enth_ref = chem_dict["enth_ref"].astype(REAL_TYPE)
        self.cp = chem_dict["cp"].astype(REAL_TYPE)
        self.pr = chem_dict["pr"].astype(REAL_TYPE)
        self.sc = chem_dict["sc"].astype(REAL_TYPE)

        self.mu_ref = chem_dict["mu_ref"].astype(REAL_TYPE)
        self.temp_ref = chem_dict["temp_ref"].astype(REAL_TYPE)

        assert self.enth_ref.shape[0] == self.num_species_full
        assert self.cp.shape[0] == self.num_species_full
        assert self.pr.shape[0] == self.num_species_full
        assert self.sc.shape[0] == self.num_species_full
        assert self.temp_ref.shape[0] == self.num_species_full
        assert self.mu_ref.shape[0] == self.num_species_full

        self.const_visc_idxs = np.squeeze(np.argwhere(self.temp_ref < 1.0e-7), axis=1)
        self.suth_visc_idxs = np.squeeze(np.argwhere(self.temp_ref >= 1.0e-7), axis=1)

        self.cp_diffs = self.cp[self.mass_frac_slice] - self.cp[-1]
        self.enth_ref_diffs = self.enth_ref[self.mass_frac_slice] - self.enth_ref[-1]

    def calc_mix_gas_constant(self, mass_fracs):
        """Compute mixture specific gas constant.

        Args:
            mass_fracs: NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.

        Returns:
            NumPy array of the mixture specific gas constant profile.
        """

        # Only need num_species mass fraction profiles
        if mass_fracs.shape[0] != self.num_species:
            mass_fracs_ns = self.get_mass_frac_array(mass_fracs_in=mass_fracs)
        else:
            mass_fracs_ns = mass_fracs

        r_mix = R_UNIV * ((1.0 / self.mol_weights[-1]) + np.sum(mass_fracs_ns * self.mw_inv_diffs[:, None], axis=0))

        return r_mix

    def calc_mix_gamma(self, mass_fracs=None, r_mix=None, cp_mix=None):
        """Compute mixture ratio of specific heats.

        Args:
            mass_fracs: NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.
            r_mix: NumPy array of the mixture specific gas constant profile.
            cp_mix: NumPy array of the mixture specific heat capacity at constant pressure profile.

        Returns:
            NumPy array of the mixture ratio of specific heats profile.
        """

        if mass_fracs is None:
            assert (r_mix is not None) and (
                cp_mix is not None
            ), "Must provide mass fractions if not providing mixture gas constant and specific heat"
        else:
            if r_mix is None:
                r_mix = self.calc_mix_gas_constant(mass_fracs)
            if cp_mix is None:
                cp_mix = self.calc_mix_cp(mass_fracs)

        gamma_mix = cp_mix / (cp_mix - r_mix)

        return gamma_mix

    def calc_mix_enth_ref(self, mass_fracs):
        """Compute mixture reference enthalpy.

        Args:
            mass_fracs: NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.

        Returns:
            NumPy array of the mixture reference enthalpy profile.
        """

        # Only need num_species mass fraction profiles
        if mass_fracs.shape[0] != self.num_species:
            mass_fracs_ns = self.get_mass_frac_array(mass_fracs_in=mass_fracs)
        else:
            mass_fracs_ns = mass_fracs

        enth_ref_mix = self.enth_ref[-1] + np.sum(mass_fracs_ns * self.enth_ref_diffs[:, None], axis=0)

        return enth_ref_mix

    def calc_mix_cp(self, mass_fracs):
        """Compute mixture specific heat at constant pressure.

        Args:
            mass_fracs: NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.

        Returns:
            NumPy array of the mixture specific heat capacity at constant pressure profile.
        """

        # Only need num_species mass fraction profiles
        if mass_fracs.shape[0] != self.num_species:
            mass_fracs_ns = self.get_mass_frac_array(mass_fracs_in=mass_fracs)
        else:
            mass_fracs_ns = mass_fracs

        cp_mix = self.cp[-1] + np.sum(mass_fracs_ns * self.cp_diffs[:, None], axis=0)

        return cp_mix

    def calc_density(self, sol_prim, r_mix=None):
        """Compute density from ideal gas law.

        Args:
            sol_prim:
                NumPy array of the primitive solution profile.
                Contains the pressure, velocity, temperature, and num_species mass fraction fields.
            r_mix:
                NumPy array of the mixture specific gas constant profile.
                If not provided, calculated from the mass fraction profiles of mass_fracs.

        Returns:

        """

        # need to calculate mixture gas constant
        if r_mix is None:
            mass_fracs = self.get_mass_frac_array(sol_prim_in=sol_prim)
            r_mix = self.calc_mix_gas_constant(mass_fracs)

        # calculate directly from ideal gas
        density = sol_prim[0, :] / (r_mix * sol_prim[2, :])

        return density

    def calc_spec_enth(self, temperature):
        """Compute individual enthalpies for all num_species_full species.

        Args:
            temperature: NumPy array of the temperature field.

        Returns:
            NumPy array of the species enthalpies profiles for all num_species_full species.
        """

        spec_enth = (
            self.cp[:, None] * np.repeat(np.reshape(temperature, (1, -1)), self.num_species_full, axis=0)
            + self.enth_ref[:, None]
        )

        return spec_enth

    def calc_stag_enth(self, velocity, mass_fracs, temperature=None, spec_enth=None):
        """Compute stagnation enthalpy.

        Args:
            velocity: NumPy array of the velocity profile.
            mass_fracs: NumPy array of mass faction profiles. Accepts num_species or num_species_full profiles.
            temperature: NumPy array of the temperature profile. Required if spec_enth is not provided.
            spec_enth: NumPy array of species enthalpies. If not provided, calculated from temperature.

        Returns:
            NumPy array of the stagnation enthalpy profile.
        """

        # get the species enthalpies if not provided
        if spec_enth is None:
            assert temperature is not None, "Must provide temperature if not providing species enthalpies"
            spec_enth = self.calc_spec_enth(temperature)

        # compute all mass fraction profiles
        if mass_fracs.shape[0] == self.num_species:
            mass_fracs = self.calc_all_mass_fracs(mass_fracs, threshold=False)

        stag_enth = np.sum(spec_enth * mass_fracs, axis=0) + 0.5 * np.square(velocity)

        return stag_enth

    def calc_species_dynamic_visc(self, temperature):
        """Compute species dynamic viscosities from Sutherland's law for all num_species_full species.

        Defaults to reference dynamic viscosity if reference temperature is zero. Otherwise computes
        species dynamic viscosities from Sutherland's law.

        Args:
            temperature: NumPy array of the temperature profile.

        Returns:
            NumPy array of the species dynamic viscosity profiles for all num_species_full species.
        """

        # TODO: should account for species-specific Sutherland temperatures

        spec_dyn_visc = np.zeros((self.num_species_full, len(temperature)), dtype=REAL_TYPE)

        # If reference temperature is (close to) zero,
        # constant dynamic viscosity
        if len(self.const_visc_idxs) > 0:
            spec_dyn_visc[self.const_visc_idxs, :] = self.mu_ref[self.const_visc_idxs, None]

        # Otherwise apply Sutherland's law
        if len(self.suth_visc_idxs) > 0:
            temp_fac = temperature[None, :] / self.temp_ref[self.suth_visc_idxs, None]
            temp_fac = np.power(temp_fac, 3.0 / 2.0)
            suth_fac = (self.temp_ref[self.suth_visc_idxs, None] + SUTH_TEMP) / (temperature[None, :] + SUTH_TEMP)
            spec_dyn_visc[self.suth_visc_idxs, :] = self.mu_ref[self.suth_visc_idxs, None] * temp_fac * suth_fac

        return spec_dyn_visc

    def calc_mix_dynamic_visc(
        self, spec_dyn_visc=None, temperature=None, mole_fracs=None, mass_fracs=None, mw_mix=None
    ):
        """Compute mixture dynamic viscosity from Wilkes mixing law.

        Args:
            spec_dyn_visc:
                NumPy array of the species dynamic viscosity profiles.
                Required if temperature is not provided.
            temperature:
                NumPy array of the temperature profile.
                Required if spec_dyn_visc is not provided.
            mole_fracs:
                NumPy array of num_species_full mole fraction profiles.
                If not provided, calculated from mass_fracs.
            mass_fracs:
                NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.
                Required if mole_fracs is not provided.
            mw_mix:
                NumPy array of the mixture molecular weight profile.
                Not required, but accelerates calculation.

        Returns:
            NumPy array of the mixture dynamic viscosity profile.
        """

        if spec_dyn_visc is None:
            assert temperature is not None, "Must provide temperature if not providing species dynamic viscosities"
            spec_dyn_visc = self.calc_species_dynamic_visc(temperature)

        if self.num_species_full == 1:

            mix_dyn_visc = np.squeeze(spec_dyn_visc)

        else:

            if mole_fracs is None:
                assert mass_fracs is not None, "Must provide mass fractions if not providing mole fractions"
                mole_fracs = self.calc_all_mole_fracs(mass_fracs, mw_mix)

            phi = np.zeros((self.num_species_full, spec_dyn_visc.shape[1]), dtype=REAL_TYPE)
            for spec_idx in range(self.num_species_full):

                muFac = np.sqrt(spec_dyn_visc[[spec_idx], :] / spec_dyn_visc)
                phi[spec_idx, :] = np.sum(
                    mole_fracs
                    * np.square(1.0 + muFac * self.mix_mass_matrix[[spec_idx], :].T)
                    * self.mix_inv_mass_matrix[[spec_idx], :].T,
                    axis=0,
                )

            mix_dyn_visc = np.sum(mole_fracs * spec_dyn_visc / phi, axis=0)

        return mix_dyn_visc

    def calc_species_therm_cond(self, spec_dyn_visc=None, temperature=None):
        """Compute species thermal conductivities for all num_species_full species.

        Args:
            spec_dyn_visc:
                NumPy array of the species dynamic viscosity profiles.
                Required if temperature is not provided.
            temperature:
                NumPy array of the temperature profile.
                Required if spec_dyn_visc is not provided.
        Returns:
            NumPy array of the species thermal conductivity coefficient profiles for all num_species_full species.
        """

        if spec_dyn_visc is None:
            assert temperature is not None, "Must provide temperature if not providing species dynamic viscosities"
            spec_dyn_visc = self.calc_species_dynamic_visc(temperature)

        spec_therm_cond = spec_dyn_visc * self.cp[:, None] / self.pr[:, None]

        return spec_therm_cond

    def calc_mix_thermal_cond(
        self, spec_therm_cond=None, spec_dyn_visc=None, temperature=None, mole_fracs=None, mass_fracs=None, mw_mix=None
    ):
        """Compute mixture thermal conductivity.

        Args:
            spec_therm_cond:
                NumPy array of species thermal conductivity profiles.
                If not provided, calculated from spec_dyn_visc or temperature.
            spec_dyn_visc:
                NumPy array of the species dynamic viscosity profiles.
                Required if spec_therm_cond and temperature are not provided.
            temperature:
                NumPy array of the temperature profile.
                Required if spec_therm_cond and spec_dyn_visc are not provided.
            mole_fracs:
                NumPy array of num_species_full mole fraction profiles.
                If not provided, calculated from mass_fracs.
            mass_fracs:
                NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.
                Required if mole_fracs is not provided.
            mw_mix:
                NumPy array of the mixture molecular weight profile.
                Not required, but accelerates calculation.

        Returns:
            NumPy array of the mixture thermal conductivity coefficient profile.
        """

        if spec_therm_cond is None:
            assert (spec_dyn_visc is not None) or (
                temperature is not None
            ), "Must provide species dynamic viscosity or temperature if not providing species thermal conductivity"

            spec_therm_cond = self.calc_species_therm_cond(spec_dyn_visc=spec_dyn_visc, temperature=temperature)

        if self.num_species_full == 1:
            mix_therm_cond = np.squeeze(spec_therm_cond)

        else:
            if mole_fracs is None:
                assert mass_fracs is not None, "Must provide mass fractions if not providing mole fractions"

                mole_fracs = self.calc_all_mole_fracs(mass_fracs, mix_mol_weight=mw_mix)

            mix_therm_cond = 0.5 * (
                np.sum(mole_fracs * spec_therm_cond, axis=0) + 1.0 / np.sum(mole_fracs / spec_therm_cond, axis=0)
            )

        return mix_therm_cond

    def calc_species_mass_diff_coeff(self, density, spec_dyn_visc=None, temperature=None):
        """Compute mass diffusivity coefficient of species into mixture for all num_species_full species.

        Args:
            density: NumPy array of the density profile.
            spec_dyn_visc:
                NumPy array of species dynamic viscosity profiles.
                If not provided, calculated from temperature.
            temperature: NumPy array of the temperature profile. Require if spec_dyn_visc is not provided.

        Returns:
            NumPy array of species mass diffusivity coefficient profiles for all num_species_full species.
        """

        # Compute species dynamic viscosities if not provided
        if spec_dyn_visc is None:
            assert temperature is not None, "Must provide temperature if not providing species dynamic viscosities"
            spec_dyn_visc = self.calc_species_dynamic_visc(temperature)

        spec_mass_diff = spec_dyn_visc / (self.sc[:, None] * density[None, :])

        return spec_mass_diff

    def calc_sound_speed(self, temperature, r_mix=None, gamma_mix=None, mass_fracs=None, cp_mix=None):
        """Compute sound speed.

        Args:
            temperature: NumPy array of the temperature profile.
            r_mix:
                NumPy array of the mixture specific gas constant profile. If not provided, calculated from mass_fracs.
            gamma_mix: NumPy array of mixture ratio of specific heats. If not provided, calculated from mass_fracs.
            mass_fracs:
                NumPy array of mass fraction profiles. Accepts num_species or num_species_full profiles.
                Required if r_mix is not provided, or if gamma_mix is not provided and cp_mix is not provided.
            cp_mix:
                NumPy array of the mixture specific heat at constant pressure profile.
                Required if gamma_mix is not provided and mass_fracs is not provided.

        Returns:
            NumPy array of sound speed profile.
        """

        # Calculate mixture gas constant if not provided
        mass_fracs_set = False
        if r_mix is None:
            assert mass_fracs is not None, "Must provide mass fractions to calculate mixture gas constant"
            mass_fracs = self.get_mass_frac_array(mass_fracs_in=mass_fracs)
            mass_fracs_set = True
            r_mix = self.calc_mix_gas_constant(mass_fracs)
        else:
            r_mix = np.squeeze(r_mix)

        # Calculate ratio of specific heats if not provided
        if gamma_mix is None:
            if cp_mix is None:
                assert mass_fracs is not None, "Must provide mass fractions to calculate mixture cp"
                if not mass_fracs_set:
                    mass_fracs = self.get_mass_frac_array(mass_fracs_in=mass_fracs)
                cp_mix = self.calc_mix_cp(mass_fracs)
            else:
                cp_mix = np.squeeze(cp_mix)

            gamma_mix = self.calc_mix_gamma(r_mix=r_mix, cp_mix=cp_mix)
        else:
            gamma_mix = np.squeeze(gamma_mix)

        sound_speed = np.sqrt(gamma_mix * r_mix * temperature)

        return sound_speed

    def calc_dens_derivs(
        self,
        density,
        wrt_press=False,
        pressure=None,
        wrt_temp=False,
        temperature=None,
        wrt_spec=False,
        mix_mol_weight=None,
        mass_fracs=None,
    ):
        """Compute derivatives of density with respect to pressure, temperature, or species mass fraction.

        For species derivatives, returns num_species derivatives. For derivation of analytical derivatives,
        please refer to the solver theory documentation.

        Args:
            density: NumPy array of density profile.
            wrt_press: Boolean flag indicating whether derivatives with respect to pressure should be computed.
            pressure: NumPy array of the pressure profile. Required if wrt_press=True.
            wrt_temp: Boolean flag indicating whether derivatives with respect to temperature should be computed.
            temperature: NumPy array of the temperature profile. Required if wrt_temp=True.
            wrt_spec:
                Boolean flag indicating whether derivatives with respect to species mass fractions should be computed.
            mix_mol_weight:
                NumPy array of the mixture molecular weight profile. If not provided, is calculated from mass_fracs.
            mass_fracs:
                NumPy array of species mass fraction profiles. Accepts num_species or num_species_full profiles.
                Required if wrt_spec=True and mix_mol_weight is not provided.

        Returns:
            Density derivative profiles w/r/t pressure, temperature, and species mass fraction, if requested.
        """

        assert any([wrt_press, wrt_temp, wrt_spec]), "Must compute at least one density derivative"

        derivs = tuple()

        # Pressure derivate
        if wrt_press:
            assert pressure is not None, "Must provide pressure for pressure derivative"
            d_dens_d_press = density / pressure
            derivs = derivs + (d_dens_d_press,)

        # Temperature derivative
        if wrt_temp:
            assert temperature is not None, "Must provide temperature for temperature derivative"
            d_dens_d_temp = -density / temperature
            derivs = derivs + (d_dens_d_temp,)

        # Species mass fraction derivatives
        if wrt_spec:
            # calculate mixture molecular weight
            if mix_mol_weight is None:
                assert mass_fracs is not None, "Must provide mass fractions to" + " calculate mixture mol weight"
                mix_mol_weight = self.calc_mix_mol_weight(mass_fracs)

            d_dens_d_mass_frac = np.zeros((self.num_species, density.shape[0]), dtype=REAL_TYPE)

            for spec_idx in range(self.num_species):
                d_dens_d_mass_frac[spec_idx, :] = (
                    density * mix_mol_weight * (1.0 / self.mol_weights[-1] - 1.0 / self.mol_weights[spec_idx])
                )

            derivs = derivs + (d_dens_d_mass_frac,)

        return derivs

    def calc_stag_enth_derivs(
        self,
        wrt_press=False,
        wrt_temp=False,
        mass_fracs=None,
        wrt_vel=False,
        velocity=None,
        wrt_spec=False,
        spec_enth=None,
        temperature=None,
    ):
        """Compute derivatives of stagnation enthalpy w/r/t pressure, temperature, velocity, or species mass fraction.

        For species derivatives, returns num_species derivatives. For derivation of analytical derivatives,
        please refer to the solver theory documentation.

        Args:
            wrt_press: Boolean flag indicating whether derivatives with respect to pressure should be computed.
            wrt_temp: Boolean flag indicating whether derivatives with respect to temperature should be computed.
            mass_fracs:
                NumPy array of species mass fraction profiles. Accepts num_species or num_species_full profiles.
                Required if wrt_temp=True.
            wrt_vel: Boolean flag indicating whether derivatives with respect to velocity should be computed.
            velocity: NumPy array of the velocity profile. Required if wrt_vel=True.
            wrt_spec:
                Boolean flag indicating whether derivatives with respect to species mass fractions should be computed.
            spec_enth: NumPy array of species enthalpies. If not provided, is calculated from temperature.
            temperature: NumPy array of temperature profile. Required if wrt_spec=True and not providing spec_enth.

        Returns:
            Stagnation enthalpy derivative profiles w/r/t pressure, temperature, velocity,
            and species mass fraction, if requested.
        """

        assert any([wrt_press, wrt_temp, wrt_vel, wrt_spec]), "Must compute at least one density derivative"

        derivs = tuple()

        # Pressure derivate
        if wrt_press:
            d_stag_enth_d_press = 0.0
            derivs = derivs + (d_stag_enth_d_press,)

        # Temperature derivative
        if wrt_temp:
            assert mass_fracs is not None, "Must provide mass fractions for temperature derivative"

            # TODO: option to provide cp
            d_stag_enth_d_temp = self.calc_mix_cp(mass_fracs)
            derivs = derivs + (d_stag_enth_d_temp,)

        # Velocity derivative
        if wrt_vel:
            assert velocity is not None, "Must provide velocity for velocity derivative"
            d_stag_enth_d_vel = velocity.copy()
            derivs = derivs + (d_stag_enth_d_vel,)

        # Species mass fraction derivative
        if wrt_spec:
            if spec_enth is None:
                assert temperature is not None, "Must provide temperature if not providing species enthalpies"
                spec_enth = self.calc_spec_enth(temperature)

            d_stag_enth_d_mass_frac = np.zeros((self.num_species, spec_enth.shape[1]), dtype=REAL_TYPE)

            if self.num_species_full == 1:
                d_stag_enth_d_mass_frac[0, :] = spec_enth[0, :]
            else:
                for spec_idx in range(self.num_species):
                    d_stag_enth_d_mass_frac[spec_idx, :] = spec_enth[spec_idx, :] - spec_enth[-1, :]

            derivs = derivs + (d_stag_enth_d_mass_frac,)

        return derivs

    def calc_press_temp_from_cons(
        self,
        density,
        total_energy,
        velocity=None,
        momentum=None,
        cp_mix=None,
        r_mix=None,
        enth_ref_mix=None,
        mass_fracs_in=None,
    ):
        """Compute temperature and pressure from a fixed total energy.

        For calorically perfect gas, this is an analytical relationship which doesn't require iteration.

        Args:
            density: NumPy array of density profile.
            total_energy: NumPy array of total energy (rhoE) profile.
            velocity: NumPy array of velocity profile. If not provided, is calculated from momentum and density.
            momentum: NumPy array of momentum profile. Required if velocity is not provided.
            cp_mix:
                NumPy array of mixture specific heat capacity at constant pressure profile.
                If not provided, is calculated from mass_fracs_in.
            r_mix:
                NumPy array of mixture specific gas constant profile.
                If not provided, is calculated from mass_fracs_in.
            enth_ref_mix:
                NumPy array of mixture reference enthalpy profile.
                If not provided, is calculated from mass_fracs_in.
            mass_fracs_in:
                NumPy array of mass fractions profile. Accepts either num_species or num_species_full profiles
                Required if cp_mix, r_mix, or enth_ref_mix are not provided.

        Returns:
            NumPy arrays of the calculated pressure and temperature profiles.
        """

        # Calculate chemical composition properties, if not provided
        mass_fracs_set = False
        if r_mix is None:
            assert mass_fracs_in is not None, "Must provide mass fractions to calculate mixture gas constant"
            mass_fracs = self.get_mass_frac_array(mass_fracs_in=mass_fracs_in)
            mass_fracs_set = True
            r_mix = self.calc_mix_gas_constant(mass_fracs)
        else:
            r_mix = np.squeeze(r_mix)

        if cp_mix is None:
            assert mass_fracs_in is not None, "Must provide mass fractions to calculate mixture cp"
            if not mass_fracs_set:
                mass_fracs = self.get_mass_frac_array(mass_fracs_in=mass_fracs_in)
            cp_mix = self.calc_mix_cp(mass_fracs)
        else:
            cp_mix = np.squeeze(cp_mix)

        if enth_ref_mix is None:
            assert mass_fracs_in is not None, "Must provide mass fractions to calculate mixture href"
            if not mass_fracs_set:
                mass_fracs = self.get_mass_frac_array(mass_fracs_in=mass_fracs_in)
            enth_ref_mix = self.calc_mix_enth_ref(mass_fracs)
        else:
            enth_ref_mix = np.squeeze(enth_ref_mix)

        # Calculate velocity, if not provided
        if velocity is None:
            assert momentum is not None, "Mst provide momentum to calculate velocity"
            velocity = momentum / density

        temp = ((total_energy / density) - np.square(velocity) / 2.0 - enth_ref_mix) / (cp_mix - r_mix)
        press = density * r_mix * temp

        return press, temp

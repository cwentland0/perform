import os
from math import sqrt

import numpy as np

from perform.solution.solution_phys import SolutionPhys
from perform.input_funcs import read_input_file
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas


# ----- BEGIN USER INPUT -----

gas_file = "~/path/to/chemistry/file.chem"

# If True, load primitive state from ic_file. If False, specify left and right primitive state
from_ic_file = False
ic_file = ""

# left and right states, if fom_ic_file = False
press_left = 1e6
vel_left = 0.0
temp_left = 300.0
mass_fracs_left = [1.0, 0.0]
press_right = 1e6
vel_right = 0.0
temp_right = 2500.0
mass_fracs_right = [0.0, 1.0]

# add bulk velocity to profile
add_vel = False
vel_add = 0.0

# ----- END USER INPUT -----

gas_file = os.path.expanduser(gas_file)

# load gas file
gas_dict = read_input_file(gas_file)
gas_model = gas_dict["gas_model"]
if gas_model == "cpg":
    gas = CaloricallyPerfectGas(gas_dict)
else:
    raise ValueError("Invalid gas_model")

# handle single-species
num_species_full = len(mass_fracs_left)
assert (
    len(mass_fracs_right) == num_species_full
), "mass_fracs_left and mass_fracs_right must have the same number of mass fractions"
assert np.sum(mass_fracs_left) == 1.0, "mass_fracs_left elements must sum to 1.0"
assert np.sum(mass_fracs_right) == 1.0, "mass_fracs_right elements must sum to 1.0"
if num_species_full == 1:
    num_species = num_species_full
else:
    num_species = num_species_full - 1
mass_fracs_slice = np.arange(num_species)

# set up states
if from_ic_file:
    sol_prim = np.load(ic_file)
    sol_prim_in = sol_prim[:, [0]]
    sol_prim_out = sol_prim[:, [-1]]

else:
    sol_prim_in = np.zeros((3 + num_species, 1), dtype=np.float64)
    sol_prim_out = np.zeros((3 + num_species, 1), dtype=np.float64)
    sol_prim_in[:3, 0] = np.array([press_left, vel_left, temp_left])
    sol_prim_in[3:, 0] = np.array(mass_fracs_left).astype(np.float64)[mass_fracs_slice]
    sol_prim_out[:3, 0] = np.array([press_right, vel_right, temp_right])
    sol_prim_out[3:, 0] = np.array(mass_fracs_right).astype(np.float64)[mass_fracs_slice]

if add_vel:
    sol_prim_in[1, 0] += vel_add
    sol_prim_out[1, 0] += vel_add

# set up solutions
sol_inlet = SolutionPhys(gas, 1, sol_prim_in=sol_prim_in)
sol_inlet.update_state(from_cons=False)
sol_outlet = SolutionPhys(gas, 1, sol_prim_in=sol_prim_out)
sol_outlet.update_state(from_cons=False)

# set some variables for ease of use
press_in = sol_inlet.sol_prim[0, 0]
vel_in = sol_inlet.sol_prim[1, 0]
temp_in = sol_inlet.sol_prim[2, 0]
rho_in = sol_inlet.sol_cons[0, 0]
cp_mix_in = sol_inlet.cp_mix[0]

press_out = sol_outlet.sol_prim[0, 0]
vel_out = sol_outlet.sol_prim[1, 0]
rho_out = sol_outlet.sol_cons[0, 0]
cp_mix_out = sol_outlet.cp_mix[0]

# calculate sound speed
c_in = gas.calc_sound_speed(
    sol_inlet.sol_prim[2, :], r_mix=sol_inlet.r_mix, mass_fracs=sol_inlet.sol_prim[3:, :], cp_mix=sol_inlet.cp_mix
)[0]
c_out = gas.calc_sound_speed(
    sol_outlet.sol_prim[2, :], r_mix=sol_outlet.r_mix, mass_fracs=sol_outlet.sol_prim[3:, :], cp_mix=sol_outlet.cp_mix
)[0]

# reference quantities
press_up = press_in + vel_in * rho_in * c_in
temp_up = temp_in + (press_up - press_in) / (rho_in * cp_mix_in)
press_back = press_out - vel_out * rho_out * c_out

# print results
# TODO: nicer string formatting
print("##### INLET #####")
print("Rho: " + str(rho_in))
print("Sound speed: " + str(c_in))
print("Cp: " + str(cp_mix_in))
print("Rho*C: " + str(rho_in * c_in))
print("Rho*Cp: " + str(rho_in * cp_mix_in))
print("Upstream pressure: " + str(press_up))
print("Upstream temp: " + str(temp_up))

print("\n")

print("##### OUTLET #####")
print("Rho: " + str(rho_out))
print("Sound speed: " + str(c_out))
print("Cp: " + str(cp_mix_out))
print("Rho*C: " + str(rho_out * c_out))
print("Rho*Cp: " + str(rho_out * cp_mix_out))
print("Downstream pressure: " + str(press_back))

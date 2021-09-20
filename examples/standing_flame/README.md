# Standing Flame

This case is similar to the contact surface in the sense that it features a cold "reactant" species and a hot "product" species. However, the viscosity and reaction are turned on for this case, with a single-step irreversible reaction mechanism which simply converts "product" to "species". Additionally, the bulk velocity of the fluid is decreased to the point that the reaction and diffusion is perfectly balanced with the bulk velocity, resulting in an effectively stationary flame. Artificial pressure forcing is applied at the outlet, causing a single-frequency acoustic wave to propagate upstream.

![Standing flame](../../doc/images/standing_flame.png)

## Sample ROM Case

A Bash script, `setup_sample_rom.sh`, is provided to download files necessary to run a sample ROM simulation for the standing flame case. Simply modify the execution permissions of the script using `chmod` and execute it in the sample case working directory. The script downloads and unpacks a slightly modified `solver_params.inp` file, a `rom_params.inp` file, the initial condition profile NumPy binary, a POD basis NumPy binary, and the necessary feature scaling profile NumPy binaries. After the script completes, simply execute "`perform .`" to test the ROM.

This sample ROM is a linear SP-LSVT ROM using a POD basis. The basis is trained from the last 2001 primitive state snapshots of the sample FOM case provided, capturing two periods of the outlet perturbation. The data is centered about the initial condition and normalized using the min-max limiter. Twenty-five basis modes are provided, though the provided `rom_params.inp` file is set up to run a five-mode simulation. As expected for this fairly simple system with one acoustic mode, the ROM accurately simulates both the training period and a future-state prediction period of equal duration.

## Balanced ROM Example

The example for balanced truncation with the eigensystem realization algorithm is a stationary flame with pressure forcing, where the balanced ROMs are trained with unit impulse response. Unit impulse is applied through the back pressure and snapshots are collected every 100 time steps. A total of 1000 Markov parameters are used to build the Hankel matrix. Balanced ROMs are then tested with sinusoidal back pressure perturbation with different amplitudes and frequencies. To adjust the testing conditions change `/inputs/brom_inputs/brom_params.inp`. 

The non-intrusive balanced truncation code operates independent of the full-order model. To test the code simply run `perform/rom/balanced_rom.py`.



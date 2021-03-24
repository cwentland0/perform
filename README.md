# **Prototyping Environment for Reacting Flow Order Reduction Methods (PERFORM)**

This code is intended to be a sort of low-fidelity Python port of the General Mesh and Equations Solver (**GEMS**), designed to allow simple implementation and testing of new reduced-order models (ROMs) for one-dimensional multi-species and reacting flow problems. Whereas **GEMS** is capable of computing high-fidelity simulations of 2D/3D reacting flows with extensive libraries for gas and flame modeling, **PERFORM** is oriented towards solving a much simpler class of 1D flames. The hope is that this code might serve as a useful testbed for ROM developers to analyze performance of new ROM methods for reacting flow simulations, a field of research which poses significant difficulties in effective model-order reduction.

## Documentation

Please see the [documentation website](perform.readthedocs.io) for a detailed user guide on installing, running, and designing new simulations for **PERFORM**. Brief descriptions of available solver routines and ROM methods are also included; please see the solver theory documentation PDF in `doc/` for more details. A very brief introduction to installing and running the code is included below. 

## Installation

Python 3.6+, `numpy`, `scipy`, and `matplotlib` are required for executing **PERFORM**. To install the package and any missing dependencies, clone this repository, enter the code's root directory `perform/`, and run the following command:

```
pip install -e .
```

This will add the script `perform` to your Python scripts directory, which is used to execute the solver.


## Running **PERFORM** 

**PERFORM** is executed using the `perform` command that was added to your Python scripts directory at installation. Execute `perform` followed by the path to the working directory of the case you wish to run. This working directory must contain a `solver_params.inp` file for the case, and a `rom_params.inp` file if you're running a ROM (both described in great detail in the documentation website). For example,

```
perform ~/path/to/working/directory
```

## Sample Cases

Three sample cases are included in `examples/`:

1. **`shock_tube`**: A non-reacting flow designed following the Sod shock tube problem, which models a diaphragm breaking between a high-pressure, high-density chamber of gas and a low-pressure, low-density chamber of gas. The resulting shock, contact surface, and rarefaction waves are a good test for the strong gradient reconstruction and future-state prediction capabilities of ROM methods.
2. **`contact_surface`**: A case that introduces strong contact surface gradients in the temperature and species mass fraction fields. There is no reaction in this case, but it serves as a good baseline for testing the ability of models to predict system acoustics in a multi-species system. By applying acoustic forcing of varying amplitudes and frequencies at the outlet, the parametric prediction capabilities of models may also be evaluated
2. **`transient_flame`**: This case follows the contact surface case, but finally introduces a single-step irreversible reaction. This case exhibits complex non-linear interactions between the flame and the system acoustics. Altering the amplitude and frequency of the outlet forcing strains the abilities of ROMs to make accurate predictions in an extremely complex parametric space.

Running one of these cases is as simple as entering its directory and executing

```
perform .
```

## Utilities

Some very simple pre/post-processing scripts are provided in `utils/`. These include scripts for generating POD basis modes, calculating input parameters for non-reflective boundary conditions etc. Brief descriptions of the scripts and their input parameters are given within the scripts.

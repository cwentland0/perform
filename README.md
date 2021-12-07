[![Documentation Status](https://readthedocs.org/projects/perform/badge/?version=latest)](https://perform.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/cwentland0/perform/badge.svg?branch=main)](https://coveralls.io/github/cwentland0/perform?branch=main)
[![DOI](https://zenodo.org/badge/281845135.svg)](https://zenodo.org/badge/latestdoi/281845135)



# **Prototyping Environment for Reacting Flow Order Reduction Methods (PERFORM)**

PERFORM is a combination 1D compressible reacting flow solver and modular reduced-order model (ROM) framework, designed to provide a simple, easy-to-use testbed for members of the ROM community to quickly prototype and test new methods on challenging (yet computationally-manageable) reacting flow problems. We hope that this tool lowers the barrier to entry for researchers from a variety of fields to develop novel ROM methods and benchmark them against an interesting set of reacting flow configurations.

## Documentation

Please see the [documentation website](https://perform.readthedocs.io) for a detailed user guide on installing, running, and designing new simulations for **PERFORM**. Brief descriptions of available solver routines and ROM methods are also included. If you would like to read the more detailed theory behind the routines, you can compile the solver theory documentation from `doc/solver_theory/main.tex` using your preferred LaTeX PDF compiler. A very brief introduction to installing and running the code is included below. 

## Installation

Python 3.6+, `numpy`, `scipy`, and `matplotlib` are required for executing **PERFORM**. Ubuntu 20.04 is the only OS that is actively tested, but MacOS users should not have many issues. To install the package and any missing dependencies, clone this repository, enter the code's root directory, and run the following command:

```
pip install -e .
```

This will add the script `perform` to your Python scripts directory, which is used to execute the solver.

## Running **PERFORM** 

**PERFORM** is executed using the `perform` command that was added to your Python scripts directory at installation. Execute `perform` followed by the path to the working directory of the case you wish to run. This working directory must contain a `solver_params.inp` file for the case, and a `rom_params.inp` file if you're running a ROM (both described in great detail in the documentation website). For example,

```
perform ~/path/to/working/directory
```

## Benchmark Cases

Four sample benchmark cases are included in `examples/`:

1. **`shock_tube`**: A non-reacting flow designed following the Sod shock tube problem, which models a diaphragm breaking between a high-pressure, high-density chamber of gas and a low-pressure, low-density chamber of gas. The resulting shock, contact surface, and rarefaction waves are a good test for the strong gradient reconstruction and future-state prediction capabilities of ROM methods.
2. **`contact_surface`**: A case that introduces strong contact surface gradients in the temperature and species mass fraction fields. There is no viscosity or reaction in this case, but it serves as a good baseline for testing the ability of models to predict system acoustics in a multi-species system. By applying acoustic forcing of varying amplitudes and frequencies at the outlet, the parametric prediction capabilities of models may also be evaluated.
3. **`standing_flame`**: This case finally introduces viscosity and a single-step irreversible reaction, albeit with a fairly simple flow regime in which the flame is stationary. The highly non-linear interaction between the flame and the system acoustics makes for challenging parametric prediction studies, especially in a ROM's ability to predict the flame's unsteady heat release rate.
4. **`transient_flame`**: The transient flame introduces a bulk velocity to the standing flame flow, causing the flame to propagate downstream. This case is a good challenge for both parametric prediction (by introducing artificial acoustic forcing) but also a ROM's future-state prediction capabilities.

Please see the `README.md` file in each sample case directory for additional details

## Testing

You can test your installation of **PERFORM** by running the included unit, integration, and regression tests. This can be done from the code's root directory by executing. 

```
chmod +x tests/run_tests.sh
tests/run_tests.sh
```
You will be prompted to choose whether to run the regression tests after the unit and integration tests have completed, as the regression tests can take a while to complete. **NOTE**: The ROM integration tests will fail if you have not installed TensorFlow 2.X; you can install TensorFlow and its dependencies with `pip install tensorflow`.

## Utilities

Some very simple pre/post-processing scripts are provided in `perform/utils/`. These include scripts for generating POD basis modes, calculating input parameters for non-reflective boundary conditions etc. Brief descriptions of the scripts and their input parameters are given within the scripts.

## Issues and Contributing

If you experience errors or unexpected solver behavior when running **PERFORM**, please first double-check your input parameters and use the [documentation](https://perform.readthedocs.io) as a reference for proper input file formatting. If problems persist, please create a new issue on this repository, and I’ll do my best to resolve it. If you would like to contribute new features or bug fixes yourself, please fork this repository and submit a pull request against this repository when you're finished.


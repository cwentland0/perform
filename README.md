# **pyGEMS** - 1D

This code is intended to be a sort of low-fidelity Python port of the General Mesh and Equations Solver (**GEMS**). Whereas **GEMS** is capable of computing high-fidelity simulations of 2D/3D reacting flows with enormous libraries for flame modeling, **pyGEMS** is intended for simulating one-dimensional reacting flows with simple global reaction mechanisms. **GEMS** was originally developed by Guoping Xia at Purdue University, and has since been expanded drastically by researchers from Purdue University, the University of Michigan, Ann Arbor, and the Air Force Research Laboratory. The hope is that this code might serve as a useful testbed for reduced-order model (ROM) developers to analyze performance of new ROM methods for reacting flow simulations, a field of research which poses significant difficulties in effective model-order reduction.

## Note to Scitech Panel Attendees

If you're visiting this page from the 2021 AIAA Scitech "Application of Data-Driven Methods to Chemically Reacting Flows" panel, please note that this project is still under development and lacks some key features. **Additionally, the documentation is grossly incomplete and out of data. It is not reflective of the current state of the code, but will be updated in short order. Please check back in the next couple days for updated documentation.** That being said, feel free to run the provided test cases and toy around with settings as you see fit. 

## Table of Contents 
* [Documentation](#documentation)
* [Installing Dependencies](#installing-dependencies)
* [Input Files](#input-files)
* [Running pyGEMS](#running-pygems)
* [Outputs](#outputs)
* [Sample Cases](#sample-cases)
* [Utilities](#utilities)

## Documentation

A very brief introduction to running **pyGEMS** is included below. However, a PDF file containing much more detailed documentation for **pyGEMS** is included in `doc/`. This contains some theory regarding the governing equations, spatial discretization, and temporal discretization of the dynamical system. The various ROM methods which are currently implemented are also explained. Tables of the various input parameters, their required data types, and default values (where applicable), are included as well. This documentation will be continuously updated as new features are added.

## Installation

Python 3.6+, `numpy`, `scipy`, and `matplotlib` are required for executing **pyGEMS**. It is only actively tested for Python 3.7 with Ubuntu 18.04. To install the package and any missing dependencies, run the following command from the top directory of **pyGEMS**:

```
pip install -e .
```

This will add the script `pygems` to your Python scripts directory, which is used to execute the solver.

## Input Files

Four input files are required compute full-order model (FOM) solutions: `solverParams.inp`, a chemistry file, a mesh file, and an initial conditions file. The `solverParams.inp` file, chemistry file, and mesh file are simple text files written by the user. The possible formats of the initial condition file are explained later. A brief explanation of each if given below:

1. **`solverParams.inp`**: This file **must** have this name, and **must** be placed in the working directory. This file defines solver (e.g. spatial/temporal discretization), unsteady output, and visualization parameters.
2. **Chemistry file**: This file can be placed anywhere and referenced in the `gasFile` parameter in `solverParams.inp`. This defines the calorically-perfect gas properties of the chemical species which are included in the system, and the Arrhenius rate parameters which govern the global reactions.
3. **Mesh file**: This file can be placed anywhere and referenced in the `meshFile` parameter in `solverParams.inp`. As of the writing of this, **pyGEMS** can only handle uniform meshes, and this file simply defines the left and right boundary coordinates and the number of finite volume cells in the discretized domain.
4. **Initial conditions file**: The definition of this file depends on its format and is explained in more detail below. This file defines the primitive field from which an unsteady simulation is initialized.

Input parameters from text input files are read by regular expressions; all must be formatted as `inputName = inputValue`, with as much white space before and after the `=` as desired. Lists should be enclosed by brackets (e.g. [val1, val2, val3]), and lists of lists should be formatted in kind (e.g. [[val11, val12],[val21, val22]]). Even if a list input only has one entry, it should be formatted as a list in the input file.

Initial condition files take one of three forms: 

1. A text file defining uniform "left" and "right" primitive states (pressure, velocity, temperature, and species mass fraction). Formatting of the parameters follows the same formatting rules for the previous text input files. Details for creating this file are given in `doc/`.
2. A NumPy `*.npy` binary file containing an array of the primitive variable fields. The leading dimension of the array should be the number of primitive fields, less one (excluding the last chemical species). The second dimension should be the number of cells in the spatial domain. Details for creating this file are given in `doc/`.
3. A restart file previously generated by **pyGEMS**. Setting the parameter `saveRestarts = True` in `solverParams.inp` will generate restart files at the interval given by `restartInterval`. When setting `initFromRestart = True` in `solverParams.inp`, **pyGEMS** will automatically restart the simulation from the most recently-saved restart file.

When running a ROM case, one additional input file is required: `romParams.inp` **must** be placed in the working directory. This input file contains information about the ROM method, projection type, model definitions, paths to model files (e.g. linear basis arrays, autoencoder binary files), normalization profile arrays, etc. 

Please see the documentation in `doc/` for detailed explanations of all possible input parameters.

## Running **pyGEMS** 

As mentioned previously, **pyGEMS** is executed using the `pygems` command that was added to your Python scripts directory at installation. Execute `pygems` followed by the path to the working directory of the case you wish to run. This working directory must contain the `solverParams.inp` file for the case, and the `romParams.inp` file if you're running a ROM. For example, 

```
pygems ~/path/to/working/directory
```

## Outputs

Upon executing **pyGEMS**, several directories will be generated in the working directory:

1. **`UnsteadyFieldResults/`**: Setting the values of `primOut`, `consOut`, and `RHSOut` to `True` will generate arrays of the time snapshots of the primitive state, conservative state, and RHS function, respectively, at the physical time step interval given by `outInterval`. 
2. **`ProbeResults/`**: Arrays containing the time history of probe measurements will be stored here. The leading dimension is the number of physical iterations in the simulation, and the second dimension is the number of variables saved plus one. The first column of this array is the physical time at each step.
3. **`ImageResults/`**: If `visSave = True`, any visualization plots will be saved here. If visualizing unsteady fields, a directory containing time snapshots of the fields will be created. If visualizing probes, single images of the entire probe time history will be written.
4. **`RestartFiles/`**: If `saveRestarts = True`, restart files will be written here at the interval specified by `restartInterval`.

## Sample Cases

Two sample cases are included in `examples/`:

1. **`shock_tube`**: A fairly simple non-reacting flow designed following the Sod shock tube problem, which models a diaphragm breaking between a high-pressure, high-density chamber of gas and a low-pressure, low-density chamber of gas. The resulting shock, contact surface, and rarefaction waves are a good test for the future-state prediction capabilities of advanced ROM methods.
2. **`contact_surface`**: A fairly simple case that introduces a strong contact surface gradients in the temperature and species mass fraction fields. There is no reaction in this case, but it serves as a good baseline for testing the ability of models to predict system acoustics in a multi-species system. By applying acoustic forcing of varying amplitudes and frequencies at the outlet, the parametric prediction capabilities of models may also be evaluated
2. **`transient_flame` (COMING SOON)**: This case follows the contact surface case, but finally introduces a premixed flame reaction. This case exhibits complex non-linear interactions between the flame and the system acoustics. Altering the amplitude and frequency of the outlet forcing strains the abilities of ROMs to make accurate predictions in an extremely complex parametric space.

After modifying the `gasFile`, `meshFile`, and `initFile`/`icParamsFile` parameters in `solverParams.inp`, running one of these cases is as simple as, e.g.,

```
pygems ./examples/shock_tube
```

## Utilities

Some very simple pre/post-processing scripts are provided in `utils/`. These include scripts for generating POD basis modes, calculating input parameters for non-reflective boundary conditions etc. Brief descriptions of the scripts and their input parameters are given within the scripts. More detailed explanations are provided in `doc/`.

## Contributing

I am actively working on making the code as modular as possible so that folks can easily integrate their flux schemes/time integrators/gas models/ROM methods into the solver. Please be patient while I clean things up and debug. If you find an error or room for optimization or better organization, feel free to make a new issue on the topic.

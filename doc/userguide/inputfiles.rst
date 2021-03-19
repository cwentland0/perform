Inputs
======
This section outlines the various input files that are required to run **PERFORM**, as well as the input parameters that are used in text input files. If you are having issues running a case (in particular, experience a ``KeyError`` error) or observing strange solver behavior, please check that all of your input parameters are set correctly, and use this page as a reference. Examples of some of these files can be found in the example cases in ``perform/examples/``.

All text input files are parsed using regular expressions. All input parameters must be formatted as ``input_name = input_value``, with as much white space before and after the equals sign as desired. A single line may contain a single input parameter definition, denoted by a single equals sign. An input file may contain as many blank lines as desired, and any line without an equals sign will be ignored (useful for user comments). Examples of various input parameters are given below

.. code-block:: python

   samp_string        = "example/string_input"
   samp_int           = 3
   samp_float_dec     = 3.14159
   samp_float_sci     = 6.02214e23
   samp_bool          = False
   samp_list          = [1.0, 2.0, 3.0]
   samp_list_of_lists = [["1_1", "1_2"],["2_1", "2_2"]]

As a rule, you should write input values as if you were writing them directly in Python code. As seen above, string values should be enclosed in double quotes (single quotes is also valid) boolean values should be written precisely as ``False`` or ``True`` (case sensitive), lists should be enclosed by brackets (e.g. ``[val_1, val_2, val_3]``), and lists of lists should be formatted in kind (e.g. ``[[val_11, val_12],[val_21, val_22]]``). Even if a list input only has one entry, it should be formatted as a list in the input file. Certain input parameters also accept a value of ``None`` (case sensitive). Each ``input_name`` is case sensitive, and string input parameters are also case sensitive. I am working on making these non-case sensitive where it's possible. 

Below, the formats and input parameters for each input file are described. For text file inputs, tables containing all possible parameters are given, along with their expected data type, default value and expected units of measurement (where applicable), and a short description of the parameter. 


.. _solverparams-label:

solver_params.inp
-----------------
The ``solver_params.inp`` file is a text file containing input parameters for running all simulations, FOM or ROM. It is the root
input file from which the gas file, mesh file, and initial condition file are specified. Further, this file specifies all parameters related to the flux scheme, time discretization, robustness control, unsteady outputs, and visualizations.

.. _meshfile-label:

Mesh File
---------
The mesh file is a text file containing input parameters for defining the computational mesh. The name and location of the mesh file is arbitrary, and is referenced from the ``mesh_file`` input parameters in ``solver_params.inp``.

As of the writing of this section, **PERFORM** can only construct uniform meshes. Thus, the defining parameters are fairly simple.


.. list-table:: Mesh file inputs
   :widths: 25 25 25 25 
   :header-rows: 1

   * - Parameter
     - Type
     - Default (Units)
     - Description
   * - x_left
     - float
     - N/A (m)
     - Spatial coordinate of inlet boundary  
   * - x_right
     - float
     - N/A (m)
     - Spatial coordinate of outlet boundary
   * - num_cells
     - int
     - N/A
     - Number of finite volume cells in mesh


Chemistry File
--------------


Piecewise Uniform IC File
-------------------------



NumPy Binary IC File
--------------------



Restart Files
-------------


.. _romparams-label:

rom_params.inp
--------------
The ``rom_params.inp`` file is a text file containing input parameters for running ROM simulations. **It must be placed in the working directory**, the same directory as its accompanying ``solver_params.inp`` file. 
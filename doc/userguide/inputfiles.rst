.. _inputs-label:

Inputs
======
This section outlines the various input files that are required to run **PERFORM**, as well as the input parameters that are used in text input files. If you are having issues running a case (in particular, experience a ``KeyError`` error) or observing strange solver behavior, please check that all of your input parameters are set correctly, and use this page as a reference. Examples of some of these files can be found in the example cases in ``perform/examples/``.

All text input files are parsed using regular expressions. All input parameters must be formatted as ``input_name = input_value``, with as much white space before and after the equals sign as desired. A single line may contain a single input parameter definition, denoted by a single equals sign. An input file may contain as many blank lines as desired, and any line without an equals sign will be ignored (useful for user comments). Examples of various input parameters are given below

.. code-block:: python

   # sample user comment
   samp_string        = "example/string_input"
   samp_int           = 3
   samp_float_dec     = 3.14159
   samp_float_sci     = 6.02214e23
   samp_bool          = False
   samp_list          = [1.0, 2.0, 3.0]
   samp_list_of_lists = [["1_1", "1_2"],["2_1", "2_2"]]

As a rule, you should write input values as if you were writing them directly in Python code. As seen above, string values should be enclosed in double quotes (single quotes is also valid), boolean values should be written precisely as ``False`` or ``True`` (case sensitive), lists should be enclosed by brackets (e.g. ``[val_1, val_2, val_3]``), and lists of lists should be formatted in kind (e.g. ``[[val_11, val_12],[val_21, val_22]]``). Even if a list input only has one entry, it should be formatted as a list in the input file. Certain input parameters also accept a value of ``None`` (case sensitive). Each ``input_name`` is case sensitive, and string input parameters are also case sensitive. We are working on making these non-case sensitive where it's possible. 

Below, the formats and input parameters for each input file are described. For text file inputs, tables containing all possible parameters are given, along with their expected data type, default value and expected units of measurement (where applicable). Note that expected types of ``list`` of ``list``\ s is abbreviated as ``lol`` for brevity. For detailed explanations of each parameter, refer to :ref:`paramindex-label`.


.. _solverparams-label:

solver_params.inp
-----------------
The ``solver_params.inp`` file is a text file containing input parameters for running all simulations, FOM or ROM. It is the root
input file from which the gas file, mesh file, and initial condition file are specified. Further, this file specifies all parameters related to the flux scheme, time discretization, robustness control, unsteady outputs, and visualizations.  **It must be placed in the working directory, and must be named** ``solver_params.inp``. Otherwise, the code will not function.


.. list-table:: ``solver_params.inp`` input parameters
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Units
   * - ``chem_file``
     - ``str``
     - \-
     - \-
   * - ``mesh_file``
     - ``str``
     - \-
     - \-
   * - ``init_file``
     - ``str``
     - \-
     - \-
   * - ``ic_params_file``
     - ``str``
     - \-
     - \-
   * - ``dt``
     - ``float``
     - \-
     - s
   * - ``time_scheme``
     - ``str``
     - \-
     - \-
   * - ``time_order``
     - ``int``
     - \-
     - \-
   * - ``num_steps``
     - ``int``
     - \-
     - \-
   * - ``subiter_max``
     - ``int``
     - ``50``
     - \-
   * - ``res_tol``
     - ``float``
     - ``1e-12``
     - Unitless
   * - ``dual_time``
     - ``bool``
     - ``True``
     - \-
   * - ``dtau``
     - ``float``
     - ``1e-5``
     - s
   * - ``adapt_dtau``
     - ``bool``
     - ``False``
     - \-
   * - ``cfl``
     - ``float``
     - ``1.0``
     - Unitless
   * - ``vnn``
     - ``float``
     - ``20.0``
     - Unitless
   * - ``run_steady``
     - ``bool``
     - ``False``
     - \-
   * - ``steady_tol``
     - ``float``
     - ``1e-12``
     - Unitless
   * - ``invisc_flux_scheme``
     - ``str``
     - ``"roe"``
     - \-
   * - ``visc_flux_scheme``
     - ``str``
     - ``"invisc"``
     - \-
   * - ``space_order``
     - ``int``
     - ``1``
     - \-
   * - ``grad_limiter``
     - ``str``
     - ``"none"``
     - \-
   * - ``bound_cond_inlet``
     - ``str``
     - \-
     - \-
   * - ``press_inlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``vel_inlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``temp_inlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``rho_inlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``mass_fracs_inlet``
     - ``list`` of ``float``
     - \-
     - BC-dependent
   * - ``pert_type_inlet``
     - ``str``
     - \-
     - \-
   * - ``pert_perc_inlet``
     - ``float``
     - \-
     - Unitless
   * - ``pert_freq_inlet``
     - ``list`` of ``float``
     - \-
     - 1/s
   * - ``bound_cond_outlet``
     - ``str``
     - \-
     - \-
   * - ``press_outlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``vel_outlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``temp_outlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``rho_outlet``
     - ``float``
     - \-
     - BC-dependent
   * - ``mass_fracs_outlet``
     - ``list`` of ``float``
     - \-
     - BC-dependent
   * - ``pert_type_outlet``
     - ``str``
     - \-
     - \-
   * - ``pert_perc_outlet``
     - ``float``
     - \-
     - Unitless
   * - ``pert_freq_outlet``
     - ``list`` of ``float``
     - \-
     - 1/s
   * - ``vel_add``
     - ``float``
     - ``0.0``
     - m/s
   * - ``stdout``
     - ``bool``
     - ``True``
     - \-
   * - ``res_norm_prim``
     - ``list`` of ``float``
     - ``[1e5, 10, 300, 1]``
     - [Pa, m/s, K, unitless]
   * - ``source_off``
     - ``bool``
     - ``False``
     - \-
   * - ``save_restarts``
     - ``bool``
     - ``False``
     - \-
   * - ``restart_interval``
     - ``int``
     - ``100``
     - \-
   * - ``num_restarts``
     - ``int``
     - ``20``
     - \-
   * - ``init_from_restart``
     - ``bool``
     - ``False``
     - \-
   * - ``probe_locs``
     - ``list`` of ``float``
     - ``[None]``
     - m
   * - ``probe_vars``
     - ``list`` of ``str``
     - ``[None]``
     - \-
   * - ``out_interval``
     - ``int``
     - ``1``
     - \-
   * - ``prim_out``
     - ``bool``
     - ``True``
     - \-
   * - ``cons_out``
     - ``bool``
     - ``False``
     - \-
   * - ``source_out``
     - ``bool``
     - ``False``
     - \-
   * - ``hr_out``
     - ``bool``
     - ``False``
     - \-
   * - ``rhs_out``
     - ``bool``
     - ``False``
     - \-
   * - ``vis_interval``
     - ``int``
     - ``1``
     - \-
   * - ``vis_show``
     - ``bool``
     - ``True``
     - \-
   * - ``vis_save``
     - ``bool``
     - ``False``
     - \-
   * - ``vis_type_X``
     - ``str``
     - \-
     - \-
   * - ``vis_var_X``
     - ``list`` of ``str``
     - \-
     - \-
   * - ``vis_x_bounds_X``
     - ``lol`` of ``float``
     - ``[[None,None]]``
     - plot-dependent
   * - ``vis_y_bounds_X``
     - ``lol`` of ``float``
     - ``[[None,None]]``
     - plot-dependent
   * - ``probe_num_X``
     - ``int``
     - \-
     - \-
   * - ``calc_rom``
     - ``bool``
     - ``False``
     - \-


.. _meshfile-label:

Mesh File
---------
The mesh file is a text file containing input parameters for defining the computational mesh. The name and location of the mesh file is arbitrary, and is referenced from the ``mesh_file`` input parameter in ``solver_params.inp``.

As of the writing of this section, **PERFORM** can solve on uniform meshes. Thus, the defining parameters are fairly simple.


.. list-table:: Mesh file inputs
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Units
   * - ``x_left``
     - ``float``
     - \-
     - m
   * - ``x_right``
     - ``float``
     - \-
     - m
   * - ``num_cells``
     - ``int``
     - \-
     - \-



.. _chemfile-label:

Chemistry File
--------------
The chemistry file is a text file containing input parameters for defining properties of the chemical species modeled in a given simulation, along with parameters which define the reactions between these species. The name and location of this file are arbitrary, and is referenced from the ``chem_file`` input parameter in ``solver_params.inp``. 

The set of parameters which is required for any gas or reaction model are given in :ref:`universalchem-label`. Those required for a calorically-perfect gas model (``gas_model = "cpg"``) are given in :ref:`cpginputs-label`. Those required for a finite-rate irreversible reaction model (``reaction_model = "fr_irrev"``) are given in :ref:`fr_irrev-label`. To be abundantly clear, **these parameters should all be given in the same chemistry file**, but they are split into different sections here for clarity.

.. _universalchem-label:

Universal Chemistry Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^
The parameters described here are required for any combination of gas model and reaction model.

.. list-table:: Universal chemistry file inputs
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Units
   * - ``gas_model``
     - ``str``
     - ``"cpg"``
     - \-
   * - ``reaction_model``
     - ``str``
     - ``"none"``
     - \-
   * - ``num_species``
     - ``int``
     - \-
     - \-
   * - ``species_names``
     - ``list`` of ``str``
     - \-
     - \-
   * - ``mol_weights``
     - ``list`` of ``float``
     - \-
     - g/mol

.. _cpginputs-label:

CPG Inputs
^^^^^^^^^^
The parameters described here are required when using a calorically-perfect gas model, i.e. when setting ``gas_model = "cpg"``.

.. list-table:: CPG chemistry file inputs
   :widths: 25 25 25 25
   :header-rows: 1
  
   * - Parameter
     - Type
     - Default
     - Units
   * - ``enth_ref``
     - ``list`` of ``float``
     - \-
     - J/kg
   * - ``cp``
     - ``list`` of ``float``
     - \-
     - J/K-kg
   * - ``pr``
     - ``list`` of ``float``
     - \-
     - Unitless
   * - ``sc``
     - ``list`` of ``float``
     - \-
     - Unitless
   * - ``temp_ref``
     - ``list`` of ``float``
     - \-
     - K
   * - ``mu_ref``
     - ``list`` of ``float``
     - \-
     - N-s/m\ :sup:`2`


.. _fr_irrev-label:

Finite Rate Irreversible Reaction Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The parameters described here are required when using a finite-rate irreversible reaction model, i.e. when setting ``reaction_model = "fr_irrev"``.


.. list-table:: Finite rate irreversible reaction model chemistry file inputs
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Units
   * - ``nu``
     - ``lol`` of ``float``
     - \-
     - Unitless
   * - ``nu_arr``
     - ``lol`` of ``float``
     - \-
     - Unitless
   * - ``act_energy``
     - ``list`` of ``float``
     - \-
     - kJ/mol
   * - ``pre_exp_fact``
     - ``list`` of ``float``
     - \-
     - Unitless
   * - ``temp_exp``
     - ``list`` of ``float``
     - \-
     - Unitless


Initial Condition Inputs
------------------------
Unsteady solutions can be initialized in three different ways in **PERFORM**: piecewise uniform function parameters files (:ref:`pwuniformfile-label`), full primitive state NumPy profiles (:ref:`npyicfile-label`), or restart files (:ref:`restartfile-label`). If multiple restart methods are requested, the following priority hierarchy is followed: restart files first, then piecewise uniform function, and finally primitive state NumPy files.


.. _pwuniformfile-label:

Piecewise Uniform IC File
^^^^^^^^^^^^^^^^^^^^^^^^^
The piecewise uniform initial condition file is a text file containing input parameters for initializing a simulation from a two-section piecewise uniform profile describing the full primitive state. This is done by specifying a "left" and "right" primitive state, and a spatial point on the computational mesh at which the two states are separated. This is ideal for initializing problems like the :ref:`sodshock-label` or flames.

.. list-table:: Piecewise uniform IC inputs
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default (Units)
     - Units
   * - ``x_split``
     - ``float``
     - \-
     - m
   * - ``press_left``
     - ``float``
     - \-
     - Pa
   * - ``vel_left``
     - ``float``
     - \-
     - m/s
   * - ``temp_left``
     - ``float``
     - \-
     - K
   * - ``mass_fracs_left``
     - ``list`` of ``float``
     - \-
     - Unitless
   * - ``press_right``
     - ``float``
     - \-
     - Pa
   * - ``vel_right``
     - ``float``
     - \-
     - m/s
   * - ``temp_right``
     - ``float``
     - \-
     - K
   * - ``mass_fracs_right``
     - ``list`` of ``float``
     - \-
     - Unitless


.. _npyicfile-label:

NumPy Primitive IC File
^^^^^^^^^^^^^^^^^^^^^^^
Providing a complete primitive state profile is by far the simplest initialization method available. The ``init_file`` parameter in ``solver_params.inp`` provides the arbitrary location of a NumPy binary (``*.npy``) containing a single NumPy array. This NumPy array must be a two- or three-dimensional array, where the first dimension is the number of governing equations in the system (3 + ``num_species`` - 1) and the second dimension is the number of cells in the discretized spatial domain. The order of the first dimension *must* be ordered by pressure, velocity, temperature, and then chemical species mass fraction. The chemical species mass fractions must be ordered as they are in the chemistry file. The optional third dimension is the time step dimension; if a higher-order time integration method is requested, the initial condition profile may provide prior time steps to preserve this order of accuracy upon initialization. If only one time step or a two-dimensional profile is provided, the time integrator will attempt to "cold start" from a first-order scheme.

This file can be generated however you like, such as ripping it manually from the unsteady outputs of a past **PERFORM** run, or generating a more complex series of discontinuous steps than what the ``ic_params_file`` settings handle natively.



.. _restartfile-label:

Restart Files
^^^^^^^^^^^^^
Restart files accomplish what the name implies: restarting the simulation from a previous point in the simulation. Restart files are saved to the ``restart_files`` directory in the working directory when ``save_restarts = True`` at an interval specified by ``restart_interval`` in ``solver_params.inp``. Two files are saved to reference a restart solution: a ``restart_iter.dat`` file and a ``restart_file_X.npz`` file, where ``X`` is the *restart iteration number*. The latter file contains both the conservative and primitive solution saved at that restart iteration, as well as the physical solution time associated with that solution. The former file is an text file containing the restart iteration number of the most recently-written restart file, and thus points to which ``restart_file_X.npz`` should be read in to initialize the solution. It is overwritten every time a restart file is written. Similarly, the maximum number of ``restart_file_X.npz`` saved to disk is dictated by ``num_restarts``. When this threshold is reached, the restart iteration number will loop back to 1 and begin overwriting old restart files.

Setting ``init_from_restart = True`` will initialize the solution from the restart file whose restart iteration number matches the one given in ``restart_iter.dat``. Thus, without modification, the solution will restart from the most recently generated restart file. However, if you want to restart from a different iteration number, you can manually change the iteration number stored in ``restart_iter.dat``.



rom_params.inp
--------------
The ``rom_params.inp`` file is a text file containing input parameters for running ROM simulations. **It must be placed in the working directory**, the same directory as its accompanying ``solver_params.inp`` file. Parameters in this file are detailed in :ref:`romparams-label`.
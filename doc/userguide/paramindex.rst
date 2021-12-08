.. _paramindex-label:

Input Parameter Index
=====================
This section provides a comprehensive index of all solver input parameters for the text input files detailed in :ref:`inputs-label`. 

solver_params.inp
-----------------
See :ref:`solverparams-label` for variable types, default values, and units (where applicable).

* ``mesh_file``: Path to mesh file. Permits absolute path, or relative path from working directory.

* ``chem_file``: Path to chemistry file. Permits absolute path, or relative path from working directory.

* ``init_file``: Path to full initial primitive state profile (stored in a ``*.npy`` NumPy binary file) to initialize the unsteady solution from. Permits absolute path, or relative path from working directory. If ``ic_params_file`` is set, or ``init_from_restart = True``, this parameter will be ignored.

* ``ic_params_file``: Path to left/right (step function) primitive state parameters file to initialize the unsteady solution from. Permits absolute path, or relative path from working directory. If ``init_from_restart = True``, this parameter will be ignored.

* ``dt``: Fixed time step size for numerical time integration.

* ``time_scheme``: Name of numerical time integration scheme to use. Please see the theory documentation for details on each scheme.

  * Valid options: ``ssp_rk3``, ``classic_rk4``, ``jameson_low_store``, ``bdf``

* ``time_order``: Order of accuracy for the chosen time integrator. Some time integrators have a fixed order of accuracy, while others may accept several different values. If a time integrator has a fixed order of accuracy and you have entered a different order of accuracy, a warning will display but execution will continue. If a time integrator accepts several values and an invalid value is entered, the solver will terminate with an error.

* ``num_steps``: Number of discrete physical time steps to run the solver through. If the solver fails before this number is reached (either through a code failure or solution blowup), any unsteady output files will be dumped to disk with the suffix ``_FAILED"`` appended to denote a failed solution.

* ``subiter_max``: The maximum number of subiterations that the iterative solver for implicit time integration schemes may execute before concluding calculations for that physical time step.

* ``res_tol``: The threshold of convergence of the :math:`\ell^2` norm of the Newton iteration residual below which the subiteration loop will automatically conclude.

* ``dual_time``: Boolean flag to specify whether dual time-stepping should be used for an implicit time integration scheme.

* ``dtau``: Fixed value of :math:`\Delta \tau` to use for dual time integration. Ignored if ``adapt_dtau = True``.

* ``adapt_dtau``:  Boolean flag to specify whether the value of :math:`\Delta \tau` should be adapted at each subiteration according to the below robustness control parameters.

* ``cfl``: Dual time-stepping Courant–Friedrichs–Lewy number to adapt :math:`\Delta \tau` based on maximum wave speed in each cell. Smaller CFL numbers will result in smaller :math:`\Delta \tau`, and greater regularization as a result.

* ``vnn``: Dual time-stepping von Neumann number to adapt :math:`\Delta \tau` based on the mixture kinematic viscosity in each cell. Smaller VNN numbers will result in smaller :math:`\Delta \tau`, and greater regularization as a result.

* ``run_steady``: Boolean flag to specify whether to run the solver in "steady" mode. See :ref:`steadymode-label` for more details.

* ``steady_tol``: If ``run_steady = True``, the threshold of convergence of the :math:`\ell^2` norm of the change in the primitive solution below which the steady solve will automatically conclude.

* ``invisc_flux_scheme``: Name of the numerical inviscid flux scheme to use. Please see the theory documentation for details on each scheme.

  * Valid options: ``roe``

* ``visc_flux_scheme``: Name of the numerical viscous flux scheme to use. Please see the theory documentation for details on each scheme.

  * Valid options: ``inviscid``, ``standard``

* ``space_order``: Order of accuracy of the state reconstructions at the cell faces for flux calculations. Must be a positive integer. If ``space_order = 1``, the cell-centered values are used. If ``space_order > 1``, finite difference stencils are used to compute cell-centered gradients, from which higher-order face reconstructions are computed. If the gradient calculation for the value entered has not been implemented, the solver with terminate with an error.

* ``grad_limiter``: Name of the gradient limiter to use when computing higher-order face reconstructions. Please see the solver documentation for details on each scheme.

  * Valid options: ``barth_cell``, ``barth_face``, ``venkat``

* ``bound_cond_inlet``: Name of the boundary condition to apply at the inlet. For details on each boundary condition, see the solver documentation. For required input parameters for a given boundary condition, see :ref:`inletbcs-label`.

  * Valid options: ``stagnation``, ``fullstate``, ``meanflow``

* ``press_inlet``: Pressure-related value for inlet boundary condition calculations.

* ``vel_inlet``: Velocity-related value for inlet boundary condition calculations.

* ``temp_inlet``: Temperature-related value for inlet boundary condition calculations.

* ``rho_inlet``: Density-related value for inlet boundary condition calculations.

* ``mass_fracs_inlet``: Chemical composition-related value for inlet boundary condition calculations.

* ``pert_type_inlet``: Type of value to be perturbed at the inlet. See :ref:`inletbcs-label` for valid options for each ``bound_cond_inlet`` value.

* ``pert_perc_inlet``: Percentage of the specified perturbed value, determining the amplitude of the inlet perturbation signal. Should be entered in decimal format, e.g. for a 10\% perturbation, enter ``pert_perc_inlet = 0.01``. See :ref:`bcpert-label` for more details.

* ``pert_freq_inlet``: List of superimposed frequencies of the inlet perturbation. See :ref:`bcpert-label` for more details.

* ``bound_cond_outlet``: Name of the boundary condition to apply at the outlet. For details on each boundary condition, see the solver documentation. For required input parameters for a given boundary condition, see :ref:`outletbcs-label`.

* ``press_outlet``: Pressure-related value for outlet boundary condition calculations.

* ``vel_outlet``: Velocity-related value for outlet boundary condition calculations.

* ``temp_outlet``: Temperature-related value for outlet boundary condition calculations.

* ``rho_outlet``: Density-related value for outlet boundary condition calculations.

* ``mass_fracs_outlet``: Chemical composition-related value for outlet boundary condition calculations.

* ``pert_type_outlet``: Type of value to be perturbed at the outlet. See :ref:`outletbcs-label` for valid options for each ``bound_cond_outlet`` value.

  * Valid options: ``subsonic``, ``meanflow``

* ``pert_perc_outlet``: Percentage of the specified perturbed value, determining the amplitude of the outlet perturbation signal. Should be entered in decimal format, e.g. for a 10\% perturbation, enter ``pert_perc_outlet = 0.1``. See :ref:`bcpert-label` for more details.

* ``pert_freq_outlet``: List of superimposed frequencies of the outlet perturbation. See :ref:`bcpert-label` for more details.

* ``vel_add``: Velocity to be added to the entire initial condition velocity field. Accepts negative values.

* ``stdout``: Boolean flag to specify whether to print iteration counts and residual norms to STDOUT.

* ``res_norm_prim``: List of values by which to normalize each field of the :math:`\ell^2` and :math:`\ell^1` residual norms before averaging across all fields. They are order by pressure, velocity, temperature, and then all species mass fractions except the last. This ensures that the norms of each residual field contribute roughly equally to the average norm used to determine Newton's method convergence.

* ``source_off``: Boolean flag to specify whether to apply the reaction source term. This is ``False`` by default; setting it manually to ``True`` turns off the source term. This can save computational cost for non-reactive cases.

* ``save_restarts``: Boolean flag to specify whether to save restart files.

* ``restart_interval``: Physical time step interval at which to save restart files.

* ``num_restarts``: Maximum number of restart files to store. After this threshold has been reached, the count returns to 1 and the first restart file is overwritten by the next restart file (and so on).

* ``init_from_restarts``: Boolean flag to determine whether to initialize the unsteady solution from

* ``probe_locs``: List of locations in the spatial domain to place point monitors. The probe measures values at the cell center closest to the specified location. If a location is less than the inlet boundary location, the inlet ghost cell will be monitored. Likewise, if a location is greater than the outlet boundary location, the outlet ghost cell will be monitored. These probe monitors are recorded at every physical time iteration and the time history is written to disk. See :ref:`probedata-label` for more details on the output. 

* ``probe_vars``: A list of fields to be probed at each specified probe location.

  * Valid for all probes: ``"pressure"``, ``"velocity"``, ``"temperature"``, ``"density"``, ``"momentum"``, ``"energy"``, ``"species_X"``, ``"density-species_X"`` (where ``X`` is replaced by the integer number of the desired chemical species to be probed, e.g. ``"species_2"`` for the second species specified in the chemistry file).
  * Valid options for interior probes only: ``"source"``, ``"heat-release"``

* ``out_interval``: Physical time step interval at which to save unsteady field data.

* ``prim_out``: Boolean flag to specify whether the unsteady primitive state should be saved.

* ``cons_out``: Boolean flag to specify whether the unsteady conservative state should be saved.

* ``source_out``: Boolean flag to specify whether the unsteady source term field should be saved.

* ``hr_out``: Boolean flag to specify whether the unsteady heat release rate should be saved.

* ``rhs_out``: Boolean flag to specify whether the unsteady right-hand-side field should be saved.

* ``vis_interval``: Physical time step interval at which to draw any requested field/probe plots. If no plots are requested, this parameter is ignored.

* ``vis_show``: Boolean flag to specify whether field/probe plots should be displayed on the user's monitor at the interval specified by ``vis_interval``. If no plots are requested, this parameter is ignored.

* ``vis_save``: Boolean flag to specify whether field/probe plots should be saved to disk at the interval specified by ``vis_interval``. If no plots are requested, this parameter is ignored.

* ``vis_type_X``: Type of data to visualize in the ``X``\ th figure. For example, ``vis_type_3`` would specify the type of the third plot to be visualized. Values of ``X`` must start from 0 and progress by one for each subsequent plot. Any gap in these numbers will cause any plots after the break to be ignored (e.g. specifying ``vis_type_0``, ``vis_type_2``, and ``vis_type_3`` without specifying ``vis_type_1`` will automatically ignore the plots for ``vis_type_2`` and ``vis_type_3``).

  * Valid options: ``field``, ``probe``

* ``probe_num_X``: 0-indexed number of the point monitor to visualize in the ``X``\ th figure if ``vis_type_X = "probe"``. Must correspond to a valid probe number.

* ``vis_var_X``: A list of fields to be plotted in the  ``X``\ th figure. Note that for ``vis_type_X = "probe"`` figures, if a specified field is not being monitored at the probe specified by ``probe_num_X``, the solver will terminate with an error.

* ``vis_x_bounds_X``: List of lists, where each sub-list corresponds to the plots specified in ``vis_var_X``. Each sublist contains two entries corresponding the lower and upper x-axis bounds for visualization of ``vis_var_X``.

* ``vis_y_bounds_X``: List of lists, where each sub-list corresponds to the plots specified in ``vis_var_X``. Each sublist contains two entries corresponding the lower and upper y-axis bounds for visualization of ``vis_var_X``.

* ``calc_rom``: Boolean flag to specify whether to run a ROM simulation. If set to ``True``, a ``rom_params.inp`` file must also be placed in the working directory. See :ref:`romparams-label` for more details on this input file.


Mesh File
---------
See :ref:`meshfile-label` for variable types, default values, and units (where applicable).

* ``x_left``: Left-most boundary coordinate of the spatial domain.  This point will be the coordinate of theleft face of the left-most finite volume cell.

* ``x_right``: Right-most boundary coordinate of the spatial domain. This point will be the coordinate of theright face of the right-most finite volume cell.

* ``num_cells``: Total number of finite volume cells in the discretized spatial domain.



Chemistry File
--------------
We break down the sections of the chemistry file input file, as in :ref:`inputs-label`.


Universal Chemistry Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^
See :ref:`universalchem-label` for variable types, default values, and units (where applicable).

* ``gas_model``: Name of the gas model to be used.

  * Valid options: ``"cpg"``

* ``reaction_model``: Name of the reaction model to be used.

  * Valid options: ``"none"``, ``"fr_irrev"``

* ``num_species``: Total number of species participating in simulation.

* ``species_names``: List of the names of the chemical species. These are only used for labeling plot axes, so they can be whatever you like (e.g. "methane", "Carbon Dioxide", "H2O"). If none are provided, these will default to ``["Species 1", "Species 2", ...]``.

* ``mol_weights``: Molecular weights of each species. Must have ``num_species`` entries.


CPG Inputs
^^^^^^^^^^
See :ref:`cpginputs-label` for variable types, default values, and units (where applicable).

* ``enth_ref``: Reference enthalpy at 0 K of each species. Must have ``num_species`` entries.

* ``cp``: Constant specific heat capacity at constant pressure for each species. Must have ``num_species`` entries.

* ``pr``: Prandtl number of each species. Must have ``num_species`` entries.

* ``sc``: Schmidt number of each species. Must have ``num_species`` entries.

* ``temp_ref``: Reference dynamic viscosity of each species for Sutherland's law. Must have ``num_species`` entries.

* ``mu_ref``: Reference temperature of each species for Sutherland's law. If ``temp_ref[i] = 0`` for any species, it will be assumed that its dynamic viscosity is constant and equal to ``mu_ref[i]``. Must have ``num_species`` entries.



Finite Rate Irreversible Reaction Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See :ref:`fr_irrev-label` for variable types, default values, and units (where applicable).

* ``nu``: List of lists of irreversible reaction stoichiometric coefficients, where each sublist corresponds to a single reaction. Reactants should have positive values, while products should have negative values.

* ``nu_arr``: List of lists of irreversible reaction molar concentration exponents for all chemical species, where each sublist corresponds to a single reaction. Those chemical species that don't participate in the reaction should just be assigned a value of ``0.0``.

* ``act_energy``: List of Arrhenius rate activation energies :math:`E_a` for each reaction.

* ``pre_exp_fact``: List of Arrhenius rate pre-exponential factors.

* ``temp_exp``: List of Arrhenius rate temperature exponents.

Piecewise Uniform IC File
-------------------------
See :ref:`pwuniformfile-label` for variable types, default values, and units (where applicable).

* ``x_split``: Location in spatial domain at which the piecewise uniform solution will be split. All cell centers with coordinates less than this value will be assigned to the "left" state, and those with coordinates greater than this value will be assigned to the "right" state.

* ``press_left``: Static pressure in "left" state.

* ``vel_left``: Velocity in "left" state.

* ``temp_left``: Temperature in "left" state.

* ``mass_fracs_left``: Species mass fractions in "left" state. Must contain ``num_species`` elements, and they must sum to 1.0.

* ``press_right``: Static pressure in "right" state.

* ``vel_right``: Velocity in "right" state.

* ``temp_right``: Temperature in "right" state.

* ``mass_fracs_right``: Species mass fractions in "right" state. Must contain ``num_species_full`` elements, and they must sum to 1.0.

rom_params.inp
--------------
See :ref:`romparams-label` for variable types, default values, and units (where applicable). We again break down some distinct sections of the file.

* ``rom_method``: Name of the ROM method to use.

  * Valid options: ``galerkin``, ``lspg``, ``mplsvt``

* ``var_mapping``: Name of the state variable mapping which the ROM models employ.

  * Valid options: ``conservative``, ``primitive``

* ``space_mapping``: Name of the mapping type which maps from the latent space to the full-order space.  

  * Valid options: ``linear``, ``autoencoder``

* ``num_models``: Number of distinct models used to make predictions for the full physical state. For example, if there is one model to predict the pressure and velocity fields, and another to predict the temperature and mass fraction fields, then ``num_models = 2``

* ``latent_dims``: A list containing the latent dimension for each model. If using a model with a fixed latent dimension (e.g. autoencoders), this will be checked against the model object and the code will terminate with an error if the values do not match

* ``model_var_idxs``: A list of lists where each sublist contains the zero-indexed state variable numbers to which each model maps. The variable numbers are ordered by density/pressure, momentum/velocity, energy/temperature, and density-weighted mass fraction/mass fraction (as ordered in the ``chem_file``). For example, in a ROM with two models, if the first model maps to velocity and mass fraction, and the second model maps to pressure and temperature, then ``model_var_idxs = [[1,3],[0,2]]``.

* ``model_dir``: Absolute path of the base under which model files and feature scaling profiles are stored.

* ``cent_ic``: Boolean flag to set ``cent_cons``/``cent_prim`` (depending on the ROM method) to the provided initial condition profile. This is simply a convenience parameter that is useful when performing parametric predictions and don't want to repeatedly change the centering profile address.

* ``norm_sub_cons``: List of paths relative to ``model_dir`` to the subtractive normalization NumPy binary profiles for feature scaling of the conservative state variables with which each model is associated. For example, if a model is associated with density/pressure and energy/temperature, then the corresponding entry in ``norm_sub_cons`` should be for the subtractive normalization profiles for the density and energy fields.

* ``norm_fac_cons``: List of paths relative to ``model_dir`` to the factor normalization NumPy binary profiles for feature scaling of the conservative state variables with which each model is associated. For example, if a model is associated with density/pressure and energy/temperature, then the corresponding entry in ``norm_fac_cons`` should be for the factor normalization profiles for the density and energy fields.

* ``cent_cons``: List of paths relative to ``model_dir`` to the centering NumPy binary profiles for feature scaling of the conservative state variables with which each model is associated. For example, if a model is associated with density/pressure and energy/temperature, then the corresponding entry in ``cent_cons`` should be for the centering profile for the density and energy fields.

* ``norm_sub_prim``: List of paths relative to ``model_dir`` to the subtractive normalization NumPy binary profiles for feature scaling of the primitive state variables with which each model is associated. For example, if a model is associated with pressure and temperature, then the corresponding entry in ``norm_sub_prim`` should be for the subtractive normalization profile for the pressure and temperature fields.

* ``norm_fac_prim``: List of paths relative to ``model_dir`` to the factor normalization NumPy binary profiles for feature scaling of the primitive state variables with which each model is associated. For example, if a model is associated with pressure and temperature, then the corresponding entry in ``norm_fac_prim`` should be for the factor normalization profile for the pressure and temperature fields.

* ``cent_prim``: List of paths relative to ``model_dir`` to the centering NumPy binary profiles for feature scaling of the primitive state variables with which each model is associated. For example, if a model is associated with pressure and temperature, then the corresponding entry in ``cent_prim`` should be for the centering profile for the pressure and temperature fields.


Linear Space Mapping Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^
See :ref:`linearinputs-label` for variable types, default values, and units (where applicable).

* ``basis_files``: List of paths relative to ``model_dir`` to the linear trial basis NumPy binary (``*.npy``) files for each model.


Autoencoder Space Mapping Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See :ref:`autoencinputs-label` for variable types, default values, and units (where applicable).

* ``decoder_files``: List of paths relative to ``model_dir`` to the decoder model objects for each model.

* ``encoder_files``: List of paths relative to ``model_dir`` to the encoder model objects for each model.

* ``decoder_isconv``: Boolean flag indicating whether the output of the decoder is a convolutional layer. If this is ``True``, then ``decoder_io_format`` must be specified.

* ``decoder_io_format``: The expected array axis ordering of the state profiles on which the decoder operates, if ``decoder_isconv = True``. See :ref:`nninputs-label` for more details.

  * Valid options: ``"channels_first"``, ``"channels_last"``

* ``encoder_isconv``: Boolean flag indicating whether the output of the encoder is a convolutional layer. If this is ``True``, then ``encoder_io_format`` must be specified.

* ``encoder_io_format``: The expected array axis ordering of the state profiles on which the encoder operates, if ``encoder_isconv = True``. See :ref:`nninputs-label` for more details.

  * Valid options: ``"channels_first"``, ``"channels_last"``

Machine Learning Library Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See :ref:`mllibinputs-label` for variable types, default values, and units (where applicable).

* ``ml_library``: Name of the machine learning library which was used to train and serialize any machine learning models to be used in the ROM.

  * Valid options: ``tfkeras``

* ``run_gpu``: Boolean flag to determine whether to run machine learning model inference on the GPU. Please note that running on the CPU is often faster than running on the GPU for these small 1D problems, as memory movement between the host and device can be extremely slow and all memory movement operations are blocking.


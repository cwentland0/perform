Outputs
=======

Unsteady Solution Data
----------------------


.. _fielddata-label:

Field Data
^^^^^^^^^^
Unsteady field data represents the time evolution of an unsteady field at the time step iteration interval specified by ``out_interval`` in ``solver_params.inp``. All unsteady field data is written to ``working_dir/unsteady_field_results``, and currently comes in four flavors: primitive state output, conservative state output, source term output, and RHS term output. All have the same general form: a NumPy binary file containing a single NumPy array with three dimensions. The first dimension is the number of variables, the second is the number of cells in the computational mesh, and the third is the number of time steps saved in the output file. The primitive state, conservative state, and RHS term output have the same number of variables, equal to the number of governing equations (3 + ``num_species`` - 1), while the source term only has (``num_species`` - 1) variables.

Primitive state field data is saved if ``prim_out = True`` and has the prefix ``sol_prim_*``. Conservative state field data is saved if ``cons_out = True`` and has the prefix ``sol_cons_*``. Source term field data is saved if ``source_out = True`` and has the prefix ``source_*``. RHS term field data is saved if ``rhs_out = True`` and has the prefix ``rhs_*``.

Unsteady field data may have three different main suffixes, depending on solver parameters: ``*_FOM`` for an unsteady full-order model simulation, ``*_steady`` for a "steady" full-order model simulation (see :ref:`steadymode-label`) for details), or ``*_ROM`` for a reduced-order model simulation. An additional suffix, ``*_FAILED``, is also appended if the solver fails (solution blowup). Thus, the conservative field results for a failed ROM run would have the name ``sol_cons_ROM_FAILED.npy``, while a successful run would simply generate ``sol_cons_ROM.npy``. 


.. _probedata-label:

Probe Data
^^^^^^^^^^
Probe/point monitor data represents the time evolution of an unsteady field variable at a single finite volume cell. Probe locations are specified by ``probe_locs`` in ``solver_params.inp``, and the fields to be measured are specified by ``probe_vars``. Valid options for ``probe_vars`` can be found in :ref:`paramindex-label`. Probe measurements are taken at every physical time step. Data for each probe is saved to a separate file in ``working_dir/probe_results``. Each has the format of a NumPy binary file containing a single two-dimensional NumPy array. The first dimension is the number of variables listed in ``probe_vars`` plus one, as the physical solution time is also stored at each iteration. The second dimension is the number of physical time steps for which the simulation was run.

The name of each probe begins with ``probe_*``, and is followed by a list of the variables stored in the probe data. Finally, the same suffixes mentioned above are applied depending on the solver settings: ``*_FOM`, ``*_steady``, and ``*_ROM``. Again, if the solver fails, the suffix ``*_FAILED`` will also be appended. Finally, the 1-indexed number of a the probe will be appended to the end of the file. For example, the second probe monitoring the velocity and momentum of a "steady" solve which fails will have the file name ``probe_velocity_momentum_2_steady_FAILED.npy``.


Restart Files
^^^^^^^^^^^^^
All restart file data is stored in ``working_dir/restart_files``. Please refer to :ref:`restartfile-label` for details on the formatting and contents of restart files.


.. _vis-label:

Visualizations
--------------
During simulation runtime, **PERFORM** is capable of generating two types of plots via ``matplotlib``: field plots and probe monitor plots. If ``vis_show = True`` in ``solver_params.inp``, then these images are displayed on the user's monitor. If ``vis_save = True``, they are saved to disk. The interval of displaying/saving the figures is given by ``vis_interval``. Each figure corresponds to a single instance of ``vis_type_X``, within which there may be several plots. Each probe figure corresponds to a single probe, from which multiple probed variables may be extracted.

All saved images are PNG images stored within ``working_dir/image_results``.

.. _fieldplot-label:

Field Plots
^^^^^^^^^^^
Field Plots display instantaneous snapshots of the entire field with the field data plotted on the y-axis and cell center coordinates plotted on the x-axis.

Field plots save an instantaneous snapshot of the field plots at the interval set by ``vis_interval``. These are stored within a subdirectory following the same pattern given to field data files, except the prefix of the directory is given by ``workding_dir/image_results/field_*``. Within this subdirectory, individual images have the prefix ``fig_*``, followed by the number of the image in the series of expected image numbers to be generated by a given run. If a simulation terminates early any field plots that were expected to be generated will not be generated.


.. _probeplot-label:

Probe Plots
^^^^^^^^^^^
Probe plots display the entire time history of the probed data up to the most recent plotting interval reached by the simulation, with the probed variable data plotted on the y-axis and time plotted on the x-axis.

A single figure is saved to disk for a given probe figure. It if first written after ``vis_interval`` time steps, after which the same file is overwritten at the interval specified by ``vis_interval``. The names of probe plots follow the same pattern given to the probe data files (except with the file extension ``*.png``, of course).
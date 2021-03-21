Miscellanea
===========
Below are details on **PERFORM** features that don't fit neatly into other documentation sections.

.. _steadymode-label:

Running in "Steady" Mode
------------------------
Setting the Boolean flag ``run_steady = True`` in ``solver_params.inp`` slightly alters the solver behavior to run in a sort of "steady-state solver" mode. To be completely clear, **PERFORM** is an unsteady solver and there are no true steady solutions for the types of problems it is designed to simulate. However, this "steady" mode is designed specifically for solving flame simulations in which bulk advection is carefully balanced with chemical diffusion and reaction forces, resulting in a roughly stationary flame, which is as close to a steady solution as one can expect for these cases. This stationary flame acts as a good "mean" flow for stationary flame problems with external forcing, or as an initial condition for transient flame problems (combined with a non-zero ``vel_add``). 

The exact changes in behavior are as follows:

* The :math:`\ell^2` and :math:`\ell^1` norms displayed in the terminal is the norm of the change in the primitive state between physical time steps. This is opposed to no residual output for explicit time integration schemes, or the linear solve residual norm for implicit time integration schemes. 

* The time history of the above residual norms will be written to the file ``unsteady_field_results/steady_convergence.dat``.

* The solver will terminate early if the :math:`\ell^2` norm of the solution change converges below the tolerance set by ``steady_tol`` in ``solver_params.inp``.

This "steady" solver can be run for both explicit and implicit time integration schemes. The procedure for obtaining "steady" flame solutions is incredibly tedious, generally requiring carefully manually tuning the boundary conditions to achieve a certain inlet velocity until a point at which the advection downstream is balanced with the diffusion and reaction moving upstream. During this tuning procedure, the user often must visually confirm that the flame is not moving by watching the field plots closely. Again, this process is incredibly tedious, but the "steady" solver helps facilitate this by providing at least one quantitative metric for determining if a steady flame solution has been achieved. 
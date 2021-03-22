.. _bounds-label:

Boundary Conditions
===================
Boundary conditions in **PERFORM** are enforced by an explicit ghost cell formulation. As noted in :ref:`solverparams-label`, the requirements and interpretations of the ``*_inlet`` and ``*_outlet`` input parameters depend on the boundary condition specified by ``bound_cond_inlet`` / ``bound_cond_outlet``. Valid options of ``pert_type`` for each boundary condition are also specified. If an invalid entry for ``pert_type`` is supplied, it will simply be ignored. For the mathematical definitions of these boundary conditions, please refer to the solver theory documentation.

In the following sections, we provide which additional input parameters are required and how they are interpreted for each valid entry of ``bound_cond_inlet``/``bound_cond_outlet``. Additionally, the acceptable values to be artificially perturbed for each boundary condition are given under ``pert_type_inlet``/``pert_type_outlet``.

.. _inletbcs-label:

Inlet BCs and Parameters
----------------------------------------

Fixed Stagnation Temperature and Pressure Inlet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This boundary condition is activated by setting ``bound_cond_inlet = "stagnation"`` in ``solver_params.inp``. This boundary specifies the upstream stagnation temperature and stagnation pressure, i.e. the temperature and pressure of the fluid when its velocity is brought to zero. Tho boundary condition results in reflections of acoustic waves and should not be used with unsteady calculations with significant system acoustics. 

The applicable boundary condition input parameters are as follows:

* ``press_inlet``: Specified stagnation pressure at the inlet.
* ``temp_inlet``: Specified stagnation temperature at the inlet.
* ``mass_fracs_inlet``: Fixed mixture composition at the inlet.


Full State Specification Inlet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This boundary condition is activated by setting ``bound_cond_inlet = "fullstate"`` in ``solver_params.inp``. This boundary conditions overspecifies the boundary condition by fixing the inlet ghost cell primitive state. This is not a useful boundary condition for unsteady calculations (unless the flow is supersonic), but is mostly useful for testing how an outlet boundary condition responds to perturbations propagating downstream. 

The applicable boundary condition input parameters are as follows:

* ``press_inlet``: Fixed static pressure at the inlet.
* ``vel_inlet``: Fixed velocity at the inlet.
* ``temp_inlet``: Fixed static temperature at the inlet.
* ``mass_fracs_inlet``: Fixed mixture composition at the inlet
* ``pert_type_inlet`` (optional): Accepts ``"pressure"``, ``"velocity"``, or ``"temperature"`` to perturb the values of the appropriate fixed quantity.

Mean Flow Inlet
^^^^^^^^^^^^^^^
This boundary condition is activated by setting ``bound_cond_inlet = "meanflow"`` in ``solver_params.inp``. This boundary condition provides a non-reflective inlet that requires some sense of a mean flow (or the flow infinitely far upstream) about which the unsteady flow is simply a perturbation. It effectively fixes the incoming characteristics while allowing the outgoing characteristics to be transmitted outside the domain without acoustic reflections. 
 
The applicable boundary condition input parameters are as follows:

* ``press_inlet``: Specified mean upstream static pressure.
* ``temp_inlet``: Specified mean upstream static temperature.
* ``mass_fracs_inlet``: Fixed mixture composition at the inlet.
* ``vel_inlet``: Specified mean upstream value of :math:`\rho c`, where :math:`c` is the sound speed.
* ``rho_inlet``: Specified mean upstream value of :math:`\rho c_p`, where :math:`c_p` is the specific heat capacity at constant pressure.
* ``pert_type_inlet`` (optional): Accepts ``"pressure"`` to perturb the mean upstream pressure.


.. _outletbcs-label:

Outlet BCs and Parameters
-----------------------------------------

Fixed Static Pressure Outlet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This boundary condition is activated by setting ``bound_cond_outlet = "subsonic"`` in ``solver_params.inp``. This boundary condition fixes the static pressure at the outlet. As with the stagnation temperature and pressure inlet, this boundary condition produces acoustic reflections at the outlet. 

The applicable boundary condition input parameters are as follows:

* ``press_outlet``: Specified static pressure at the outlet.
* ``mass_frac_outlet``: Fixed mixture composition at the outlet.
* ``pert_type_outlet`` (optional): Accepts ``"pressure"`` to perturb the pressure at the outlet.

Mean Flow Outlet
^^^^^^^^^^^^^^^^
This boundary condition is activated by setting ``bound_cond_outlet = "meanflow"`` in ``solver_params.inp``. This boundary condition, as with the mean flow inlet boundary condition, fixes the incoming characteristic while transmitting the outgoing characteristics without reflections. Again, it requires some sense of a mean flow (or the flow infinitely far downstream) about which the unsteady flow is simply a perturbation.

The applicable boundary condition input parameters are as follows:

* ``press_outlet``: Specified mean downstream static pressure.
* ``vel_outlet``: Specified mean downstream value of :math:`\rho c`, where :math:`c` is the sound speed.
* ``rho_outlet``: Specified mean downstream value of :math:`\rho c_p`, where :math:`c_p` is the specific heat capacity at constant pressure.
* ``pert_type_outlet`` (optional): Accepts ``"pressure"`` to perturb the mean downstream pressure.


  .. _bcpert-label:

Boundary Perturbations
----------------------

Setting valid values for ``pert_type_inlet`` or ``pert_type_outlet``, as well as non-zero values of ``pert_perc_inlet`` / ``pert_freq_inlet`` or ``pert_perc_outlet`` / ``pert_freq_outlet``, initiates external forcing at the appropriate boundary. The perturbation signal is a simple sinusoid, given for a given perturbed quantity :math:`\alpha` in the boundary ghost cell as

.. math::

    \alpha(t) = \overline{\alpha} \left(1 + A \sum_{i=1}^{N_f}\text{sin}(2 \pi f_i t) \right)


where :math:`\overline{\alpha}` is the relevant reference quantity given in ``solver_params.inp``, :math:`f_i` are the signal frequencies in ``pert_freq_inlet``/``pert_freq_outlet``, and :math:`A` is the amplitude percentage ``pert_perc_inlet``/``pert_perc_outlet``. 

For example, if the user sets (among other required parameters)

.. code-block:: python

	bound_cond_outlet = "meanflow"
	press_outlet      = 1.0e6
	pert_type_outlet  = "pressure"
	pert_perc_outlet  = 0.05
	pert_freq         = [2000.0, 5000.0]

this will result in two perturbation signals (one of 2 kHz, another of 5 kHz) of the mean downstream static pressure with amplitude 50 kPa. 
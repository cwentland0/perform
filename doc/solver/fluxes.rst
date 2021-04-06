.. _fluxschemes-label:

Flux Schemes
============
This section outlines the various schemes available for computing the inviscid and viscous fluxes of the 1D Navier-Stokes equations with species transport. For details on the mathematics of each scheme, please refer to the theory documentation.

Inviscid Flux Schemes
---------------------

Roe Scheme
^^^^^^^^^^
This inviscid flux scheme is activated by setting ``invisc_flux_scheme = "roe"`` in ``solver_params.inp``. The Roe scheme follows the approximate Riemann solver of Philip Roe :cite:p:`1981:roeflux`. As of the writing of this section, the code is not capable of applying an entropy fix for locally-sonic flows, but will be available in a forthcoming release.


Viscous Flux Schemes
--------------------

Inviscid Scheme
^^^^^^^^^^^^^^^
This viscous flux scheme is activated by setting ``visc_flux_scheme = "invisc"`` in ``solver_params.inp``. This scheme simply neglects all contributions from the viscous flux terms.

Standard Viscous Scheme
^^^^^^^^^^^^^^^^^^^^^^^
This viscous flux scheme is activated by setting ``visc_flux_scheme = "standard"`` in ``solver_params.inp``. This scheme computes the contribution from the viscous fluxes terms from the average state at each cell face (in the case of the Roe flux, the Roe average) and face gradients computed by a second-order accurate finite difference stencil. The viscous fluxes are then computed directly, with the sole approximation of the diffusion velocity term given by

.. math::

	V_l Y_l = -D_{l, M} \frac{\partial Y_l}{\partial x}

where :math:`D_{l, M}` is the mass diffusion coefficient for the :math:`l`\ th species diffusing into the mixture. This approximation is inserted into both the heat flux and species transport viscous flux term. Please see the theory documentation for calculation of the diffusion coefficients

The calculation of individual species diffusion velocities is incredibly expensive, necessitating this approximation. However, this may lead to violations of mass conservation in solving the species transport equations. A correction velocity term is included which helps mitigate this, and is described in detail in the theory documentation.
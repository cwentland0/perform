.. _goveqs-label:

Governing Equations
===================
**PERFORM** solves the 1D Navier-Stokes equations with chemical species transport and a chemical reaction source term. This can be formulated as

.. math::
   \frac{\partial \mathbf{q}}{\partial t} + \frac{\partial}{\partial x}(\mathbf{f} - \mathbf{f}_v) = \mathbf{s}

.. math::
   \mathbf{q} = 
   \begin{bmatrix}
      \rho \\ \rho u \\ \rho h^0 - p \\ \rho Y_l
   \end{bmatrix}, \quad
   \mathbf{f} = 
   \begin{bmatrix}
      \rho u \\  \rho u^2 + p \\ \rho h^0 u \\ \rho Y_l
   \end{bmatrix}, \quad
   \mathbf{f}_v = 
   \begin{bmatrix}
      0 \\ \tau \\ u \tau - q \\ -\rho V_l Y_l
   \end{bmatrix}, \quad
   \mathbf{s} = 
   \begin{bmatrix}
        0 \\ 0 \\ 0 \\ \dot{\omega}_l
   \end{bmatrix}

where :math:`\mathbf{q}` is the conservative state, :math:`\mathbf{f}` is the inviscid flux vector, :math:`\mathbf{f}_v` is the viscous flux vector, and :math:`\mathbf{s}` is the source term. Additionally, :math:`\rho` is density, :math:`u` is velocity, :math:`h^0` is stagnation enthalpy, :math:`p` is static pressure, and :math:`Y_l` is the mass fraction of the :math:`l`\ th chemical species. For a system with :math:`N_Y` chemical species, only :math:`N_Y - 1` species transport equations are solved, as the final species mass fraction :math:`Y_{N_Y}` can be computed from the fact that all mass fractions must sum to unity.

The spatial domain is discretized by the finite volume method. Brief notes on available flux calculations schemes are given in :ref:`fluxschemes-label`. Descriptions of gradient limiters for higher-order schemes are given in :ref:`gradlimiters-label`. Available boundary conditions are described in :ref:`bounds-label`. Available numerical time integrators are detailed in :ref:`timeschemes-label`.

Some details on available gas models for calculating relevant thermodynamic and transport properties are given in :ref:`gasmodels-label`. Models for calculating the source term :math:`\dot{\omega}_l` are detailed in :ref:`reacmodels-label`.

Details on each of these topics are provided in the theory documentation. Additionally, those interested in how this theory may be extended to higher dimensions and to more complex gas/reaction models are directed to Matthew Harvazinski's thesis :cite:p:`Harvazinski2012`. This details the inner mechanics of **GEMS**, the high-fidelity 3D combusting flow solver which **PERFORM**'s baseline solver is based off of.
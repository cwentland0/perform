.. _gasmodels-label:

Gas Models
==========
This section describes the models available for describing the thermodynamic and transport properties of gases. As of the writing of this section, there are no plans to include real gas models, and so all models will make the perfect gas assumption. This means that all intermolecular forces are neglected, and the gas is governed by the ideal gas law

.. math::

   p = \rho R T
	
The two gas models which are planned to be included in **PERFORM** differ in how thermodynamic properties (e.g. enthalpy, entropy) and transport properties (e.g. dynamic viscosity, diffusion coefficients) are calculated.

Calorically-perfect Gas
-----------------------
The calorically perfect gas model is activated by setting ``gas_model = "cpg"`` in the ``chem_file``. The CPG model assumes that the heat capacity at constant pressure for each species is constant, i.e. :math:`c_{p,l}(T) = c_{p,l}`. These values are given in the ``chem_file`` via ``cp``. The species enthalpies are thus given by

.. math::
   h_l = h_{ref,l} + c_{p,l} T

where the reference enthalpies at 0 K are given in the ``chem_file`` via ``enth_ref``.

Species dynamic viscosities are computed via Sutherland's law,

.. math::
   \mu_l (T) = \mu_{ref, l} \left( \frac{T}{T_{ref, l}} \right)^{3/2} \left( \frac{T_{ref, l} + S}{T + S} \right)

where the species reference temperatures are given in the ``chem_file`` via ``temp_ref``, and the reference viscosities via ``mu_ref``. The Sutherland temperature is given as :math:`S = 110.4` K. If :math:`T = 0` K, then :math:`\mu_l = \mu_{ref,l}`.

The species thermal conductivities (required for calculating the heat flux) are given by

.. math::
   K_l = \frac{\mu_l c_{p,l}}{\text{Pr}_l}

Where the species Prandtl numbers :math:`\text{Pr}_l` are given in the ``chem_file`` via ``pr``. The binary diffusion coefficients of each species into the mixture is given by

.. math::
   D_{l,M} = \frac{\mu_l}{\rho \text{Sc}_l}

where the species Schmidt number :math:`\text{Sc}_l` are given in the ``chem_file`` via ``sc``.

Additional details on the CPG gas model, particularly on computing the mixture thermodynamic and transport properties, can be found in the theory documentation.

Thermally-perfect Gas
---------------------
Coming soon!
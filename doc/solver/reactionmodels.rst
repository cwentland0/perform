.. _reacmodels-label:

Reaction Models
===============
This section briefly describes the reaction models available in **PERFORM**. For a comprehensive description of combustion mechanics well beyond the reaction models covered here (though also including them), we direct the reader to *Combustion* by Glassman and Yetter :cite:p:`Glassman2008`.

Finite-rate Mechanisms
----------------------
Finite-rate reactions generally describe mixtures in chemical non-equilibrium, opposed to infinitely fast chemistry in which reactions are assumed to proceed to completion instantaneously. Finite-rate mechanisms must, as the name implies, compute finite reaction rates. 

To preface this, we begin with the general form of the :math:`m`\ th reaction (in a set of :math:`N_r` reactions) between :math:`N_Y` chemical species, given by

.. math::
   \sum_{l=1}^{N_Y} \nu_{l,m}' \chi_l \overset{k_{r,m}}{\underset{k_{f,m}}{\leftrightharpoons}} \sum_{l=1}^{N_Y} \nu_{l,m}'' \chi_l

where :math:`\chi_l` is the chemical formula for the :math:`l`\ th chemical species, and :math:`\nu_{l,m}'` and :math:`\nu_{l,m}''` are the stoichiometric coefficients of the reactants and products, respectively. These stoichiometric coefficients are input into **PERFORM** as the difference between the reactant coefficient and the product coefficient, i.e. :math:`\nu_{l,m} = \nu_{l,m}' - \nu_{l,m}''`, via ``nu`` in the ``chem_file``.

The coefficients :math:`k_{f,m}` and :math:`k_{r,m}` are the forward and reverse reaction rates for the :math:`m`\ th reaction. The forward reaction rate is computed as an Arrhenius rate, given by the formula

.. math::
   k_{f,m} = A_m T^{b_m} \text{exp} \left( \frac{-E_{a,m}}{R_u T} \right)

where the coefficients :math:`A_m`, :math:`b_m`, and :math:`E_{a,m}` are tabulated constants given by the reaction mechanism, given by ``pre_exp_fact``, ``temp_exp``, and ``act_energy``, respectively, in the ``chem_file``.

The reaction source term :math:`\dot{\omega}_l` introduced in :ref:`goveqs-label` is computed as a function of reaction "rates of progress" :math:`w_m`

.. math::
   \dot{\omega}_l = W_l\sum_{m=1}^{N_r} (\nu_{l,m}'' - \nu_{l,m}') w_m

where :math:`W_l` is the molecular weight of the :math:`l`\ th species. The following methods are concerned with the calculation of these rates of progress.


Irreversible Mechanisms
^^^^^^^^^^^^^^^^^^^^^^^
The irreversible reaction mechanism model is activated by setting ``reaction_model = "fr_irrev"`` in the ``chem_file``. An irreversible finite-rate mechanism assumes that reactions only proceed in the forward direction, i.e. converting reactants to products and neglecting the reverse reaction rate :math:`k_{r,m}`. The rate of progress for the :math:`m`\ th reaction is given by

.. math::
   w_m = k_{f,m} \prod_{l=1}^{N_Y} [X_l]^{\tilde{\nu}_{l,m}}

where :math:`[X_l]` is the molar concentration of the :math:`l`\ th species. Additionally, :math:`\tilde{\nu}_{l,m}` are tabulated constants for each species and reaction which are input in the ``chem_file`` via ``nu_arr``.

Irreversible reactions vastly simplify the calculation of the reaction source term, at the expense of accuracy. The exponential constants :math:`\tilde{\nu}_{l,m}` are empirically-determined and may not be accurate under all flow and reaction regimes. The reduced cost of these mechanisms is often extremely attractive, and errors incurred by their approximations may be within acceptable limits.

Reversible Mechanisms
^^^^^^^^^^^^^^^^^^^^^
Coming soon!
.. _linearsubroms-label:

Linear Subspace Projection ROMs
===============================

We begin describing linear projection ROMs by defining a general non-linear ODE which governs our dynamical system, given by

.. math::

   \frac{d \mathbf{q}}{dt} = \mathbf{R}(\mathbf{q})

where for ODEs describing conservation laws, :math:`\mathbf{q} \in \mathbb{R}^N` is the conservative state, and the non-linear right-hand side (RHS) term :math:`\mathbf{R}(\mathbf{q})` is the spatial discretization of fluxes, source terms, and body forces. For linear subspace ROMs, we make an approximate representation of the system state via a linear combination of basis vectors,

.. math:: 

   \mathbf{q} \approx \widetilde{\mathbf{q}} = \overline{\mathbf{q}} + \mathbf{P} \sum_{i=1}^K \mathbf{v}_i \widehat{q}_i =  \overline{\mathbf{q}} + \mathbf{P} \mathbf{V} \widehat{\mathbf{q}}

The basis :math:`\mathbf{V} \in \mathbb{R}^{N \times K}` is referred to as the "trial basis", and the vector :math:`\widehat{\mathbf{q}} \in \mathbb{R}^K` are the generalized coordinates. The matrix :math:`\mathbf{P}` is simply a constant diagonal matrix which scales the model prediction. :math:`K`, sometimes referred to as the "latent dimension", is chosen such that :math:`K \ll N`. By far the most popular means of computing the trial basis is the proper orthogonal decomposition method.

Inserting this approximation into the FOM ODE, projecting the governing equations via the "test" basis :math:`W \in \mathbb{R}^{N \times K}`, and rearranging terms arrives at 

.. math::
   \frac{d \widehat{\mathbf{q}}}{dt} = \left[\mathbf{W}^T  \mathbf{V} \right]^{-1} \mathbf{W}^T \mathbf{P}^{-1} \mathbf{R}\left( \widetilde{\mathbf{q}} \right) 

This is now a :math:`K`\ -dimensional ODE which may be evolved with any desired time integration scheme. However, for general non-linear ODEs, it is unlikely that any cost reduction is actually achieved, as the majority of the computational cost for sufficiently complex.

The following sections provide brief details on how various linear subspace projection ROMs are formulated in relation to the above ROM formulation.

Galerkin Projection
-------------------
The linear Galerkin projection ROM is activated by setting ``rom_method = "linear_galerkin_proj"``. As the name implies, this method applies Galerkin projection by selecting :math:`\mathbf{W} = \mathbf{V}`. If :math:`\mathbf{V}` is an orthonormal basis, the ROM formulation simplifies to 

.. math::
   \frac{d \widehat{\mathbf{q}}}{dt} = \mathbf{V}^T \mathbf{P}^{-1} \mathbf{R} \left( \widetilde{\mathbf{q}} \right) 

Although Galerkin projection ROMs have been extensively studied and can be successful when applied to fairly simple fluid flow problems, they exhibit very poor accuracy and stability for practical flows. 

This method requires setting the ``cent_cons``, ``norm_sub_cons``, and ``norm_fac_cons`` feature scaling profiles in ``rom_params.inp``.

**NOTE**: Galerkin ROMs target the conservative variables, and the trial bases input via ``model_files`` should be trained as such. If you attempt to run the simulation with dual time-stepping (``dual_time = True``) it will terminate with an error.

LSPG Projection
---------------
The linear least-squares Petrov-Galerkin (LSPG) projection ROM is activated by setting ``rom_method = "linear_lspg_proj"``. This method is so named because it is derived by solving the non-linear least-square problem

.. math::

   \widehat{\mathbf{q}} = \underset{\mathbf{a} \in \mathbb{R}^K}{argmin} || \mathbf{P}^{-1} \mathbf{r} \left( \overline{\mathbf{q}} + \mathbf{P} \mathbf{V} \mathbf{a} \right) ||_2^2

where :math:`\mathbf{r}()` is the fully-discrete residual, i.e. the set of equations arising from discretizing the FOM ODE in time. Solving this problem via Gauss-Newton, the :math:`s`\ th subiteration is given by

.. math::

   \left(\mathbf{W}^s\right)^T \mathbf{W}^s (\widehat{\mathbf{q}}^{s+1} - \widehat{\mathbf{q}}^{s}) = -\left(\mathbf{W}^s\right)^T \mathbf{P}^{-1} \mathbf{r}\left( \widetilde{\mathbf{q}}^s \right)

where 

.. math::

   \mathbf{W}^s = \mathbf{P}^{-1} \frac{\partial \mathbf{r}\left( \widetilde{\mathbf{q}}^s\right)}{\partial \widetilde{\mathbf{q}}} \mathbf{P} \mathbf{V}

In general, LSPG has been shown to produce more stable and accurate ROMs than Galerkin ROMs for a given number of trial modes. However, LSPG ROMs are significantly more computationally expensive (requiring the calculation of a time-variant test basis which involves the residual Jacobian). Further, LSPG ROMs deteriorate to a Galerkin projection ROM when using an explicit time integrator or as :math:`\Delta t \rightarrow 0`. If you attempt to run an LSPG ROM with an explicit time integrator, the code will terminate with an error.

This method requires setting the ``cent_cons``, ``norm_sub_cons``, and ``norm_fac_cons`` feature scaling profiles in ``rom_params.inp``.

**NOTE**: LSPG ROMs, as with Galerkin ROMs, target the conservative variables, and the trial bases input via ``model_files`` should be trained as such. If you attempt to run the simulation with dual time-stepping (``dual_time = True``) it will terminate with an error.

SP-LSVT Projection
------------------
The linear structure-preserving least-squares with variable transformations (SP-LSVT) projection ROM is activated by setting ``rom_method = "linear_splsvt_proj"``. This method leverages :ref:`dualtime-label` to allow the trial bases to target an arbitrary (but complete) set of solution variables, instead of the conservative variables. This is particularly useful for combustion problems, where we would like to work with the primitive variables. To begin, the method proposes a similar representation of the primitive state as a linear combination of basis vectors

.. math:: 

   \mathbf{q}_p \approx \widetilde{\mathbf{q}}_p = \overline{\mathbf{q}}_p + \mathbf{H} \sum_{i=1}^K \mathbf{v}_{p,i} \widehat{q}_{p,i} =  \overline{\mathbf{q}}_p + \mathbf{H} \mathbf{V}_p \widehat{\mathbf{q}}_p

where :math:`\mathbf{V}_p` and :math:`\widehat{\mathbf{q}}_p` are the trial basis and generalized coordinates for the primitive variable representation. Here, :math:`\mathbf{H}` is a constant diagonal scaling matrix for the primitive state. Similar to LSPG, SP-LSVT solves the non-linear least-squares problem

.. math::

   \widehat{\mathbf{q}}_p = \underset{\mathbf{a} \in \mathbb{R}^K}{argmin} || \mathbf{P}^{-1} \mathbf{r}_{\tau} \left( \overline{\mathbf{q}}_p + \mathbf{H} \mathbf{V}_p \mathbf{a} \right) ||_2^2

where :math:`\mathbf{r}_\tau()` is the fully-discrete *dual-time* residual. Solving this problem via Gauss-Newton, the :math:`s`\ th subiteration is given by

.. math::

   \left(\mathbf{W}^s\right)^T \mathbf{W}^s (\widehat{\mathbf{q}}_p^{s+1} - \widehat{\mathbf{q}}_p^{s}) = -\left(\mathbf{W}^s\right)^T \mathbf{P}^{-1} \mathbf{r}_{\tau} \left( \widetilde{\mathbf{q}}_p^s \right)

where 

.. math::

   \mathbf{W}^s = \mathbf{P}^{-1} \frac{\partial \mathbf{r}_{\tau}\left( \widetilde{\mathbf{q}}_p^s\right)}{\partial \widetilde{\mathbf{q}}_p} \mathbf{H} \mathbf{V}_p

SP-LSVT is quite similar to LSPG, but has shown exceptional accuracy and stability improvements over LSPG for combustion problems. 

This method requires setting the ``cent_prim``, ``norm_sub_prim``, ``norm_fac_prim``, and ``norm_fac_cons`` feature scaling profiles in ``rom_params.inp``.

**NOTE**: SP-LSVT ROMs target the primitive variables, and the trial bases input via ``model_files`` should be trained as such. If you attempt to run the simulation without dual time-stepping (``dual_time = False``) it will terminate with an error.
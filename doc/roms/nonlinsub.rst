.. _nonlinsubroms-label:

Non-linear Subspace Projection ROMs
===================================

Over the past decade, it has become increasingly clear that linear subspace ROMs, i.e. those that represent the solution as a linear combination of trial basis vectors, are severely lacking when applied to practical fluid flow problems. Their difficulty in reconstructing sharp gradients and their inability to generalize well beyond the training dataset call into question whether they can be a useful tool for parametric or future-state prediction. This idea is synthesized by the concept of the Kolmogorov n-width,

.. math::

   d_n(\mathcal{A}, \mathcal{X}) \triangleq \underset{\mathcal{X}_n}{\vphantom{sup}\text{inf}} \enspace \underset{x \in \mathcal{A}}{\text{sup}} \enspace \underset{y \in \mathcal{X}_n}{\vphantom{sup}\text{inf}} || x - y ||_{\mathcal{X}}

which measures how well a subset :math:`\mathcal{A}` of a space :math:`\mathcal{X}` can be represented by an :math:`n`\ -dimensional subspace :math:`\mathcal{X}_n`. Those subsets for which an increase in :math:`n` does not improve the representation much are said to have a "slowly-decaying" n-width. The solution of advection-dominated flows, which characterize most practical engineering systems, have a slowly-decaying n-width, and as such a linear representation of the solution may be quite poor.

Non-linear representations of the solution, and the ROM methods which arise from them, seek to overcome this problem. The solution approximation can be recast in a more general form as

.. math:: 

   \mathbf{q} \approx \widetilde{\mathbf{q}} = \overline{\mathbf{q}} + \mathbf{P} \mathbf{g}\left(\widehat{\mathbf{q}}\right)

where :math:`\mathbf{g}: \mathbb{R}^K \rightarrow \mathbb{R}^N` is some non-linear mapping from the low-dimensional state to the physical full-dimensional state. In theory, the non-linear solution manifold to which the decoder maps can more accurately represent the governing ODE solution manifold.

A particularly attractive option for developing this non-linear mapping is from autoencoders, an unsupervised learning neural network architecture. This class of neural networks attempts to learn the identity mapping by ingesting full-dimensional state data, "encoding" this to a low-dimensional state (the "code"), and then attempting to "decode" this back to the original full-dimensional state data.  After the network is trained, the "decoder" half of the network is used as the non-linear mapping :math:`\mathbf{g}` in the ROM. 

This approach has seen exceptional success for fairly simple advection-dominated problems, but is still in its infancy and has yet to be tested for any practical problems. However, it is not without its drawbacks. The cost of evaluating the neural network decoder (and its Jacobian, as will be seen later) greatly exceeds the cost of computing the linear "decoding" :math:`\mathbf{V} \widehat{\mathbf{q}}`. The decoder predictions are also prone to noisy predictions even in regions of smooth flow. Although work is being done in developing graph neural networks, the traditional convolutional autoencoders can only be applied to solutions defined on Cartesion meshes. Further, the neural network is a black box model with no true sense of optimality besides "low" training and validation error.

The implementation of these non-linear autoencoder ROMs is dependent on the software library used to train the neural network, e.g. TensorFlow or PyTorch. Some details on how these neural networks should be formatted and input to **PERFORM** are given in :ref:`tfkeras-inputs`.


Manifold Galerkin Projection
----------------------------
The non-linear autoencoder manifold Galerkin projection ROM with TensorFlow-Keras neural networks is activated by setting ``rom_method = "autoencoder_galerkin_proj_tfkeras"``. After inserting the approximate state into the FOM ODE, this method leverages the chain rule and rearranges terms to arrive at 

.. math::

   \frac{d \widehat{\mathbf{q}}}{dt} = \mathbf{J}_d^+ \left( \widehat{\mathbf{q}} \right) \mathbf{P}^{-1} \mathbf{R}(\widetilde{\mathbf{q}})

where :math:`\mathbf{J}_d \triangleq \partial \mathbf{g}/ \partial \widehat{\mathbf{q}}: \mathbb{R}^K \rightarrow \mathbb{R}^{N \times K}` is the Jacobian of the decoder. 

This method has been shown to greatly outperform linear Galerkin projection ROMs for simple advection-dominated flow problems, immensely improving shock resolution and parametric prediction at extremely low :math:`K`. Unsurprisingly though, the computational cost of this method is much, much greater than its linear subspace counterpart.

This method requires setting the ``cent_cons``, ``norm_sub_cons``, and ``norm_fac_cons`` feature scaling profiles in ``rom_params.inp``.

**NOTE**: Manifold Galerkin ROMs target the conservative variables, and the encoders/decoders input via ``encoder_files``/``model_files``, respectively, should be trained as such. If you attempt to run the simulation with dual time-stepping (``dual_time = True``) it will terminate with an error.


.. _encoderform-label:

Encoder Jacobian Form
^^^^^^^^^^^^^^^^^^^^^
Due to the exceptional cost of this method, an approximate method has been proposed to at least circumvent the cost of computing the pseudo-inverse of the decoder Jacobian. Under some generous assumptions of negligible error in the autoencoding procedure, we can approximate

.. math::

   \mathbf{J}_d^+ \left( \widehat{\mathbf{q}} \right) \approx \mathbf{J}_e \left( \widetilde{\mathbf{q}} \right)

where :math:`\mathbf{J}_e \triangleq \partial \mathbf{h}/ \partial \widetilde{\mathbf{q}}: \mathbb{R}^N \rightarrow \mathbb{R}^{K \times N}` is the Jacobian of the *encoder* half of the autoencoder, :math:`\mathbf{h}(\widetilde{\mathbf{q}})`. Unfortunately, this method is only applicable to Galerkin projection ROMs using *explicit* time integrators, as implicit time integration requires both the decoder Jacobian and its pseudo-inverse, eliminating the usefulness of this substitution. This method has yet to be demonstrated successfully on practical problems, but has had some success for the parametrized 1D Burgers' equation.

The encoder Jacobian form for manifold Galerkin ROMs with explicit time integrators is activated by setting ``encoder_jacob = True`` in ``rom_params.inp``. Of course, the encoders must be provided via ``encoder_files``.


Manifold LSPG Projection
------------------------
The non-linear autoencoder manifold least-squares Petrov-Galerkin (LSPG) projection ROM with TensorFlow-Keras neural networks is activated by setting ``rom_method = "autoencoder_lspg_proj_tfkeras"``. The method follows the same procedure as the linear equivalent, but the resulting test basis takes the form

.. math::

   \mathbf{W}^s = \mathbf{P}^{-1} \frac{\partial \mathbf{r}\left( \widetilde{\mathbf{q}}^s\right)}{\partial \widetilde{\mathbf{q}}} \mathbf{P} \mathbf{J}_d(\widehat{\mathbf{q}}^s)

Some results indicate that manifold LSPG ROMs are more accurate than manifold Galerkin ROMs for a given number of trial modes. However, as with the linear ROMs, manifold LSPG is significantly more computationally expensive and still deteriorates to manifold Galerkin projection when using an explicit time integrator or as :math:`\Delta t \rightarrow 0`. If you attempt to run an LSPG ROM with an explicit time integrator, the code will terminate with an error.

This method requires setting the ``cent_cons``, ``norm_sub_cons``, and ``norm_fac_cons`` feature scaling profiles in ``rom_params.inp``.

**NOTE**: Manifold LSPG ROMs, as with manifold Galerkin ROMs, target the conservative variables, and the encoders/decoders input via ``encoder_files``/``model_files``, respectively, should be trained as such. If you attempt to run the simulation with dual time-stepping (``dual_time = True``) it will terminate with an error.


SP-LSVT Projection
------------------
The non-linear autoencoder manifold structure-preserving least-squares with variable transformations (SP-LSVT) projection ROM with TensorFlow-Keras neural networks is activated by setting ``rom_method = "autoencoder_splsvt_proj_tfkeras"``. As with its linear counterpart, the manifold SP-LSVT begins by providing a representation of the *primitive* state

.. math:: 

   \mathbf{q}_p \approx \widetilde{\mathbf{q}}_p =  \overline{\mathbf{q}}_p + \mathbf{H} \mathbf{g}_p \left( \widehat{\mathbf{q}}_p \right)

Again following the same dual-time residual minimization procedure arrives at a similar test basis of the form

.. math::

   \mathbf{W}^s = \mathbf{P}^{-1} \frac{\partial \mathbf{r}_{\tau}\left( \widetilde{\mathbf{q}}_p^s\right)}{\partial \widetilde{\mathbf{q}}_p} \mathbf{H} \mathbf{J}_{d,p} \left( \widehat{\mathbf{q}}_p^s \right)

Again, although manifold SP-LSVT is quite similar to manifold LSPG, early results indicate that it is much more accurate and stable than manifold LSPG for combustion problems. 

This method requires setting the ``cent_prim``, ``norm_sub_prim``, ``norm_fac_prim``, and ``norm_fac_cons`` feature scaling profiles in ``rom_params.inp``.
   
**NOTE**: Manifold SP-LSVT ROMs target the primitive variables, and the encoders/decoders input via ``encoder_files``/``model_files``, respectively, should be trained as such. If you attempt to run the simulation without dual time-stepping (``dual_time = False``) it will terminate with an error.
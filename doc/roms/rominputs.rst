.. _rominputs-label:

ROM Input Files
===============
This section outlines the various input files that are required to run ROMs in **PERFORM**, as well as the input parameters that are used in text input files. If you are having issues running a case (in particular, experiencing a ``KeyError`` error), please check that all of your input parameters are set correctly, and use this page as a reference. Text files inputs should be formatted exactly as described in :ref:`inputs-label`.

Below, the formats and input parameters for each input file are described. For text file inputs, tables containing all possible parameters are given, along with their expected data type, default value and expected units of measurement (where applicable). Note that expected types of ``list`` of ``list``\ s is abbreviated as ``lol`` for brevity. For detailed explanations of each parameter, refer to :ref:`paramindex-label`.

.. _romparams-label:

rom_params.inp
--------------
The ``rom_params.inp`` file is a text file containing input parameters for running ROM simulations. It specifies all parameters related to the ROM model and number of models, the latent dimension of each model, and the paths to model input files and standardization profiles.  **It must be placed in the working directory alongside** ``solver_params.inp`` **, and must be named** ``rom_params.inp``. Otherwise, the code will not function.

The table below provides input parameters which may be required by any ROM method. Input parameters which are specific to the neural network autoencoder ROMs are given in :ref:`autoencinputs-label`.

.. list-table:: ``rom_params.inp`` input parameters
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Units
   * - ``rom_method``
     - ``str``
     - \-
     - \-
   * - ``num_models``
     - ``int``
     - \-
     - \-
   * - ``latent_dims``
     - ``list`` of ``int``
     - ``[0]``
     - \-
   * - ``model_var_idxs``
     - ``lol`` of ``int``
     - ``[[-1]]``
     - \-
   * - ``model_dir``
     - ``str``
     - \-
     - \-
   * - ``model_files``
     - ``list`` of ``str``
     - \-
     - \-
   * - ``cent_ic``
     - ``bool``
     - ``False``
     - \-
   * - ``norm_sub_cons``
     - ``list`` of ``str``
     - ``[""]``
     - \-
   * - ``norm_fac_cons``
     - ``list`` of ``str``
     - ``[""]``
     - \-
   * - ``cent_cons``
     - ``list`` of ``str``
     - ``[""]``
     - \-
   * - ``norm_sub_prim``
     - ``list`` of ``str``
     - ``[""]``
     - \-
   * - ``norm_fac_prim``
     - ``list`` of ``str``
     - ``[""]``
     - \-
   * - ``cent_prim``
     - ``list`` of ``str``
     - ``[""]``
     - \-

.. _autoencinputs-label:

Autoencoder ROM Inputs
^^^^^^^^^^^^^^^^^^^^^^
The parameters described here may be used in ``rom_params.inp`` when applying a neural network autoencoder ROM.

.. list-table:: Autoencoder ROM input parameters
   :widths: 25 25 25 25
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Units
   * - ``encoder_files``
     - ``list`` of ``str``
     - \-
     - \-
   * - ``io_format``
     - ``str``
     - \-
     - \-
   * - ``encoder_jacob``
     - ``bool``
     - ``False``
     - \-
   * - ``run_gpu``
     - ``bool``
     - ``False``
     - \-


Feature Scaling Profiles
------------------------
Feature scaling is a routine procedure in data science for ensuring that the datasets used to train a model are normalized and no specific feature if given an inordinate amount of weight in the training procedure. This addresses the wide range of magnitudes seeing in flow field data: pressure can be :math:`\mathcal{O}(1\text{e}6)`, temperature can be :math:`\mathcal{O}(1\text{e}3)`, velocity can be :math:`\mathcal{O}(10)`, and species mass fraction is :math:`\mathcal{O}(1)`. When data is ingested by the model or the model makes a prediction during the inference stage (e.g. ROM runtime), the same scaling procedure must be applied.

In **PERFORM**, ROM models are generally trained on and operate on snapshots of the conservative or primitive state profile. Data standardization of a solution profile (given here generally by :math:`\mathbf{u}`) is computed as

.. math::

   \mathbf{u}' = \frac{\mathbf{u} - \mathbf{u}_{cent} - \mathbf{u}_{sub}}{\mathbf{u}_{fac}}

We refer to :math:`\mathbf{u}_{cent}` as the "centering" profile, :math:`\mathbf{u}_{sub}` as the "subtractive" normalization profile, and :math:`\mathbf{u}_{fac}` as the "factor" normalization profile. The reverse procedure, de-scaling, is simply given by

.. math::

   \mathbf{u} = \mathbf{u}' \odot \mathbf{u}_{fac} + \mathbf{u}_{cent} + \mathbf{u}_{sub}

Conservative and primitive state centering profiles are input via ``cent_cons`` and ``cent_prim`` in ``rom_params.inp``, respectively. Conservative and primitive subtractive normalization profiles are input via ``norm_sub_cons`` and ``norm_sub_prim`` in ``rom_params.inp``, respectively. Finally, the conservative and primitive factor normalization profiles are input via ``norm_fac_cons`` and ``norm_fac_prim`` in ``rom_params.inp``, respectively.

It may seem strange to separate :math:`\mathbf{u}_{cent}` and :math:`\mathbf{u}_{sub}`, as their repeated summation would simply be wasted FLOPS. Indeed, under the hood these profiles are summed and treated as a single profile at runtime. However, during the pre-processing stage it is generally easier for the user to treat these separate. For example, the centering profile may be the time-averaged mean profile or initial condition profile, while the normalization profiles may come from min-max scaling of the centered data. We thus allow the user this flexibility in deciding how to express these profiles.


Model Objects
-------------
We operate under the assumption that every ROM method provides some mapping from a low-dimensional representation of the state to the physical full-dimensional state, sometimes referred to as a "decoder." We generalize this mapping to allow for multiple decoders which may map to a subset of the state variables, each with their own low-dimensional state. For example, a ROM method may provide two decoders, one which predicts the pressure and velocity fields, and another which predicts the temperature and species mass fraction fields. In various contexts this has been referred to as a "scalar" or "separate" ROM. The more traditional method of using a single decoder for the entire full-dimensional state, with only one low-dimensional state vector, is sometimes referred to as a "vector" or "coupled" ROM. 

The total number of models is given by the ``num_models`` parameter in ``rom_params.inp``, and the dimension of each model's low-dimensional state is given by each entry in ``latent_dims``. The zero-indexed state variables to which each model maps is given by each sublist in ``model_var_idxs``. The model object(s) required for this decoding procedure is located via each entry in ``model_files``.


Linear Bases
^^^^^^^^^^^^
For ROM models which require a linear basis representation (such as those described in :ref:`linearsubroms-label`), each model object located by ``model_files`` in ``rom_params.inp`` is a three-dimensional NumPy binary (``*.npy``) containing the linear trial basis for that model. The first dimension is the number of state variables that the trial basis represents, the second dimension is the number of cells in the computational domain, and the third dimension is the number of trial modes generated by the basis calculation procedure. This final dimension is the *maximum* number of trial modes which may be requested via the corresponding entry in ``latent_dims``.

.. _nninputs-label:

Neural Networks
^^^^^^^^^^^^^^^
The model objects for neural network-based ROMs are generally specific to each network training framework (e.g. Keras, PyTorch). In general, they are serialized as a single file when saved to disk and can be deserialized at runtime.

The expected format in which the neural networks interact with field data is given by ``io_format`` in ``rom_params.inp``. As of the writing of this section, the only valid options are ``"nchw"`` and ``"nhwc"``. The former indicates that the neural network operates with field data arrays whose first dimension is the batch size, the second dimension is the number of state variables ("channels"), and the final channel is the spatial dimension. The latter swaps the channel dimension and spatial dimension ordering. 

.. _tfkeras-inputs:

TensorFlow-Keras Autoencoders
"""""""""""""""""""""""""""""
TensorFloat-Keras autoencoders must be serialized separately as an encoder and a decoder via the ``model.save()`` function. As of the writing of this section, only the older Keras HDF5 format (``*.h5``) can be loaded by **PERFORM**, but support for the newer TensorFlow SavedModel format should be along shortly. The decoder files are located via ``model_files`` in ``rom_params.inp``, while the encoder files (which are only required when initializing the low-dimensional solution from the full-state solution or when ``encoder_jacob = True``) are located via ``encoder_files``.

**NOTE**: if running with ``run_gpu = False`` (making model inferences on the CPU), note that TensorFlow convolutional layers cannot handle a ``channels_first`` format. If your network format conforms to ``io_format = "nchw"``, the code will terminate with an error. This issue could theoretically be fixed by the user by including a permute layer to change the layer input ordering to ``channels_last`` before any convolutional layers, but we err on the side of caution here.
.. _quickstart-label:

Quick Start
===========

Dependencies
------------

**PERFORM** is a pure Python code and does not (as of the writing of this section) depend on any non-Python software. As such, everything can be installed directly through ``pip`` and is done so through the ``pip install`` command explained below.

The code is only actively tested using Python 3.7.4. I have arbitrarily set the minimum required Python version to be 3.6. I also only actively test it in Ubuntu 18.04, but it should run without a hitch on other Linux distributions, Windows, and macOS.

The baseline solver only requires three additional packages: ``numpy``, ``scipy``, and ``matplotlib``. I have not set minimum required version for these packages, but will when I make some time to check at what version the code fails.

The TensorFlow-Keras autoencoder ROM models of course depend on ``tensorflow``, though the code will only throw an error if you attempt to run one of those models and so does not require ``tensorflow`` to run the baseline solver or other ROM models. These models are only tested for ``tensorflow==2.4.1``, and I make no guarantees whatsoever that the code will work correctly or optimally for older versions.

Installing
----------
To get up and running with **PERFORM**, first clone the source code repository by executing the following command from your terminal command line:

.. code-block:: bash

   git clone https://github.com/cwentland0/perform.git

or use your Git client of choice, along with the above HTTPS address. **PERFORM** is currently installed locally via ``pip`` (or ``pip3``, if your ``pip`` does not automatically install for your Python 3 distribution). To do this, enter the **PERFORM** root folder and execute

.. code-block:: bash

    pip install -e .

This will install **PERFORM**, as well as any required package dependencies which you have not yet installed.

Running
-------
After installation is complete, a new script, ``perform``, will be added to you Python scripts. This is the command that you will use to execute **PERFORM**. The options for running the code from the command line are as follows:

.. code-block:: bash
   perform [-h] [-w WORK] [-p PARAM] [-r ROM]
      -w, --work WORK    runtime working directory
      -p, --param PARAM  solver parameters input file path
      -r, --rom ROM      ROM parameter input file path

Note that all command-line options are optional: ``WORK`` defaults to the terminal's current working directory, ``PARAM`` defaults to ``./solver_params.inp``, and ``ROM`` defaults to ``./rom_params.inp``. The format and input key values of ``PARAM`` and ``ROM`` are described in :ref:`solverparams-label` and :ref:`romparams-label`, respectively.

You can check that **PERFORM** works by entering any of the example case directories (e.g. ``perform/examples/shock_tube``) and executing

.. code-block:: bash

   perform

If running correctly, your terminal's STDOUT should start filling with output from each time step iteration and live field and probe plots should appear on your screen.

Testing
-------
Coming soon!
.. _quickstart-label:

Quick Start
===========

Dependencies
------------

**PERFORM** is a pure Python code and does not (as of the writing of this section) depend on any non-Python software. As such, everything can be installed directly through ``pip`` and is done so through the ``pip install`` command explained below. 

The minimum required Python version is 3.6, but is only actively tested with Python 3.8. Additionally, it is only actively test in Ubuntu 20.04, but should run without many problems on other Linux distributions, Windows Subsystem for Linux (WSL), and macOS. Some issues have been noted in non-Ubuntu OS's when executing Bash scripts used for downloading data from Google Drive and executing tests. 

The baseline solver only requires three additional packages: ``numpy``, ``scipy``, and ``matplotlib``. These will be installed, or updated if your local installations are older than the minimum required versions, upon installing **PERFORM**.

Neural network ROM models using TensorFlow-Keras of course depend on ``tensorflow``, though the code will only throw an error if you attempt to run one of those models and so does not require ``tensorflow`` to run the baseline solver or other ROM models. These models are only tested for the most recent production release of TensorFlow 2, and we make no guarantees that the code will work correctly or optimally for older versions (and will definitely not work with TensorFlow 1).

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
After installation is complete, a new script, ``perform``, will be added to you Python scripts. This is the command that you will use to execute **PERFORM**, followed by the path to the working directory of the case you would like to run, e.g.

.. code-block:: bash

   perform /path/to/working/directory

**The working directory of a case is dictated by the presence of a** ``solver_params.inp`` **file**, described in detail in :ref:`solverparams-label`. The code will not execute if there is not a properly-formatted ``solver_params.inp`` file in the specified working directory. **If you are running a ROM case, an additional** ``rom_params.inp`` **file must also be placed in the working directory**. This file is described in detail in :ref:`romparams-label`.

You can check that **PERFORM** works by entering any of the example case directories (e.g. ``perform/examples/shock_tube``) and executing

.. code-block:: bash

   perform .

If running correctly, your terminal's STDOUT should start filling with output from each time step iteration and live field and probe plots should appear on your screen. Alternatively, you can run the test suite described below to check that your installation works as expected.


Testing
-------
A suite of unit, integration, and regression tests are included in ``perform/tests/``. These can be run manually from the **PERFORM** root directory by executing

.. code-block:: bash

   tests/run_tests.sh

This will automatically run unit and integration tests and report whether they succeeded or failed. You will then be prompted as to whether you would like to run the regression tests. These can take a while to complete, and really only needs to be checked before submitting a pull request. Note that the regression tests use the included example cases, so if you altered the input files for those cases then make sure to reset them before executing the regression tests.  
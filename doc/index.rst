PERFORM
===========
**PERFORM** is a combination 1D compressible reacting flow solver and modular reduced-order model (ROM) framework, designed to provide a simple, easy-to-use testbed for members of the ROM community to quickly prototype and test new methods on challenging (yet computationally-manageable) reacting flow problems. The baseline solver is based off the **General Equations and Mesh Solver (GEMS)**:cite:p:`Ding2004`, a legacy Fortran 3D reacting flow solver originally developed by Li Ding and Guoping Xia at Purdue University. This code has obviously been simplified to one-dimensional flows, but it aims to be a) open-source, freely available to copy and modify for everyone, and b) much more approachable for researchers outside the high-performance computing community.  We hope that this tool lowers the barrier to entry for researchers from a variety of fields to develop novel ROM methods and benchmark them against an interesting set of reacting flow configurations.

This documentation serves as a reference for users to get up and running with **PERFORM**, understand the underlying solver (albeit very superficially), and become acquainted with implementing new ROM routines and evaluating their performance. The mathematical theory behind the solver and ROM methods is only touched on briefly; we refer the interested reader to the PDF document supplied in ``perform/doc/PERFORM_solver_doc.pdf`` for more details. This website and the theory documentation is very much a work-in-progress and will be updated regularly.

Data-driven Modeling for Complex Fluid Physics
---------------------------------------------------------
This code also serves as a companion code for the workshop on *Data-driven Modeling for Complex Fluid Physics*, presented at the 2021 AIAA SciTech Forum. Details on the workshop and several proposed benchmark cases can be found at the `workshop website <https://sites.google.com/umich.edu/romworkshop/home>`_. The proposed benchmark cases, among others, are provided in ``perform/examples/``. Those interested in keeping up-to-date on workshop activities can send an email to ``romworkshop [at] gmail [dot] com`` requesting to be added to the workshop mailing list.

Acknowledgements
----------------
Christopher R. Wentland acknowledges support from the US Air Force Office of Scientific Research through the Center of Excellence Grant FA9550-17-1-0195 (Technical Monitors: Fariba Fahroo, Mitat Birkan, Ramakanth Munipalli, Venkateswaran Sankaran). This work has also been supported by a grant of computer time from the DOD High Performance Computing Modernization Program.

.. toctree::
   :maxdepth: 3
   :caption: User Guide
   :hidden:

   userguide/quickstart
   userguide/examples
   userguide/inputfiles
   userguide/outputfiles
   userguide/paramindex
   userguide/misc
   userguide/contributions

.. toctree::
   :maxdepth: 3
   :caption: Solver
   :hidden:

   solver/goveqs
   solver/fluxes
   solver/limiters
   solver/bound
   solver/timeschemes
   solver/gasmodels
   solver/reactionmodels
   

.. toctree::
	:maxdepth: 3
	:caption: ROMs
	:hidden:

	roms/rommodel
	roms/rominputs
	roms/linearsub
	roms/nonlinsub
	roms/liftlearn
	roms/caetcn

.. toctree::
	:maxdepth: 3
	:caption: Extras
	:hidden:

	extras/license
	extras/references

# Transient Flame

This case exhibits all of the features of the previous multi-species test cases: a cold "reactant" species diffusing into a hot "product" species, a single-step reaction mechanism, and a higher bulk fluid velocity to cause the flame to advect downstream. The sharp gradients in temperature and species mass fraction, the stiff reaction source term, and the bulk advection of the sharp gradients make for a fairly challenging problem.

![Unforced transient flame](../../doc/images/transient_flame_without_forcing.png)

As with the contact surface, the complexity of the transient flame problem may be further increased by applying artificial pressure forcing at the outlet, causing an acoustic wave to propagate upstream. As the amplitude and frequency of the forcing is increased, the interaction between the system acoustics and the flame becomes increasingly complex.

![Forced transient flame](../../doc/images/transient_flame_with_forcing.png)
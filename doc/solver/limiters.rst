.. _gradlimiters-label:

Gradient Limiters
=================
For higher-order face reconstructions with a fixed stencil, gradient limiters are required to prevent excessive oscillations near sharp gradients. When initializing simulations from a strong step function (``ic_params_file``), a gradient limiter is practically required to ensure simulation stability (at least for the first few time steps).

Barth-Jespersen Limiter
-----------------------
This limiter guarantees that no new local maxima or minima are created by the higher-order face reconstructions. However, the gradient limiter calculation is non-differentiable (due to a minimum function), which can negatively affect convergence of the solver. 

Venkatakrishnan Limiter
-----------------------
The Venkatakrishnan limiter improves on the Barth-Jespersen limiter by replacing the non-differentiable minimum function with a smooth polynomial function. This has the effect of improving solver convergence, but has the negative consequence of limiting the solution in smooth regions and more aggressively smoothing discontinuities.
.. _timeschemes-label:

Time Integrators
================

This section briefly describes the various numerical time integrators that are available in **PERFORM**. For details on each scheme, please refer to the theory documentation.



Explicit Integrators
--------------------
Explicit time integrators depend only on prior time steps, and thus no not require the solution of a linear system. As there is no concept of a linear solve residual as in the iterative solution of implicit schemes, **PERFORM** will simply report the time step iteration number in the terminal. Explicit schemes are generally less stable than implicit schemes and require smaller time steps, especially for well-resolved combustion problems. However, given the relative cost of the implicit solve Jacobian calculations and linear system solution in **PERFORM**, you can sometimes achieve a cheaper solution with an explicit scheme with a smaller time step over an implicit scheme with a larger time step.

Classic RK4
^^^^^^^^^^^
The classic RK4 scheme is activated by setting ``time_scheme = "classic_rk4"`` in ``solver_params.inp``. This is the classic fourth-order accurate explicit Runge-Kutta scheme, originally proposed by Martin Kutta in 1901. The Butcher tableau for this scheme is given below.

.. math::
   \begin{array}
   {c|cccc}
   0 & 0 & 0 & 0 & 0 \\
   1/2 & 1/2 & 0 & 0 & 0 \\
   1/2 & 0 & 1/2 & 0 & 0 \\
   1 & 0 & 0 & 1 & 0 \\
   \hline
   & 1/6 & 1/3 & 1/3 & 1/6 
   \end{array}

Strong Stability-preserving RK3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The strong stability-preserving RK3 (SSPRK3) scheme is activated by setting ``time_scheme = "ssp_rk3"`` in ``solver_params.inp``. Strong stability-preserving methods are named in reference to preserving the strong stability properties of the forward Euler scheme while providing higher orders of accuracy. This scheme provides such a third-order accurate scheme. The Butcher tableau for this scheme is given below.

.. math::
   \begin{array}
   {c|ccc}
   0 & 0 & 0 & 0 \\
   1 & 1 & 0 & 0 \\
   1/2 & 1/4 & 1/4 & 0 \\
   \hline
   & 1/6 & 1/6 & 2/3  
   \end{array}

Jameson Low-storage Scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Jameson low-storage scheme by setting ``time_scheme = "jameson_low_store"`` in ``solver_params.inp``. What we refer to as the "Jameson low-storage scheme" is simply a scheme presented by Jameson which, while only appropriate for steady calculations, is a vast simplification over of the typical Runge-Kutta methods and extensible to arbitrary orders of accuracy. For an :math:`s`\ -stage RK scheme, the :math:`i`\ th stage calculation is given by

.. math::

   q^i = q^n - \frac{\Delta t}{s + 1 - i} f(q^{i-1})

Thus, the scheme only requires the solution at the previous time step and the RHS evaluation at the previous time step, greatly decreasing the storage cost of the scheme.

Implicit Integrators
--------------------
Implicit time integrators depend on the system solution at future time steps, and thus can only be solved approximately. **PERFORM** uses Newton's method to iteratively solve the fully-discrete system. Newton's method is repeatedly applied until the :math:`\ell^2` norm of the linear solve residual converges below the threshold given by ``res_tol``, or ``subiter_max`` iterations are computed. Implicit time integrators generally exhibit excellent stability properties at relatively large time steps. As such, they are well-suited to combustion problems, which are typically extremely stiff due to the strong exponential non-linearity arising from the reaction source term. However, the cost of computing the Jacobian of the RHS function and solving the stiff linear system can be quite expensive relative to the cost of computing the RHS side for simple 1D problems.

Backwards Differentiation Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Backwards differentiation formula (BDF) schemes are activated by setting ``time_scheme = "bdf"`` in ``solver_params.inp``. BDF schemes are a particular class of linear multi-step schemes. As of the writing of this section, **PERFORM** is capable of computing the first-order, second-order, third-order, and fourth-order accurate BDF schemes. However, it is not advised to use anything higher than the second-order accurate scheme (``time_order = 2``), as these higher-order accurate schemes can sometimes be unstable.

.. _dualtime-label:

Dual Time-stepping
------------------
Dual time-stepping :cite:p:`Venkateswaran1995` is activated by using implicit time integration scheme and setting ``dual_time = True`` in ``solver_params.inp``. This method is a time integration method which adds a pseudo-time derivative to the governing equations,

.. math::
   \Gamma \frac{\partial \mathbf{q}_p}{\partial \tau} + \frac{\partial \mathbf{q}}{\partial t} + \frac{\partial}{\partial x}(\mathbf{f} - \mathbf{f}_v) = \mathbf{s}

where :math:`\tau` is the pseudo-time variable, :math:`\mathbf{q}_p = [p \; u \; T \; Y_l]^\top` are the primitive variables, and :math:`\Gamma = \partial \mathbf{q} / \partial \mathbf{q}_p`. Numerical integration of these equations with an implicit solver has two beneficial effects: the pseudo-time term has the effect of regularizing the linear solve, improving its stability, and the primitive state :math:`\mathbf{q}_p` can be solved for directly, instead of the conservative state :math:`\mathbf{q}`. This latter point is particularly key for reacting systems, as computing the primitive state from the conservative state can be extremely challenging when using a thermally-perfect or real gas model.
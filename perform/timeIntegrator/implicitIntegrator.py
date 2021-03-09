import numpy as np

import perform.constants as const
from perform.inputFuncs import catch_input
from perform.time_integrator.time_integrator import TimeIntegrator

class ImplicitIntegrator(TimeIntegrator):
	"""
	Base class for implicit time integrators
	Solves implicit system via Newton's method
	"""

	def __init__(self, param_dict):
		super().__init__(param_dict)
		self.time_type      = "implicit"
		self.subiter_max    = catch_input(param_dict, "subiter_max", const.subiter_max_imp_default)
		self.res_tol        = catch_input(param_dict, "res_tol", const.l2_res_tol_default)

		# Dual time-stepping, robustness controls
		self.dual_time      = catch_input(param_dict, "dual_time", True)
		self.dtau           = catch_input(param_dict, "dtau", const.dtauDefault)
		if (self.dual_time):
			self.adapt_dtau = catch_input(param_dict, "adapt_dtau", False)
		else:
			self.adapt_dtau = False
		self.cfl            = catch_input(param_dict, "cfl", const.cfl_default)   # reference CFL number for advective control of dtau
		self.vnn            = catch_input(param_dict, "vnn", const.vnn_default)   # von Neumann number for diffusion control of dtau
		self.ref_const      = catch_input(param_dict, "ref_const", [None])        # constants for limiting dtau	
		self.relax_const    = catch_input(param_dict, "relax_const", [None])      #


class BDF(ImplicitIntegrator):
	"""
	Backwards difference formula (up to fourth-order)
	"""

	def __init__(self, param_dict):
		super().__init__(param_dict)

		self.coeffs = [None]*4
		self.coeffs[0] = np.array([1.0, -1.0], dtype=const.real_type)
		self.coeffs[1] = np.array([1.5, -2.0, 0.5], dtype=const.real_type)
		self.coeffs[2] = np.array([11./16., -3.0, 1.5, -1./3.], dtype=const.real_type)
		self.coeffs[3] = np.array([25./12., -4.0, 3.0, -4./3., 0.25], dtype=const.real_type)
		assert (self.time_order <= 4), str(self.time_order)+"th-order accurate scheme not implemented for "+self.time_scheme+" scheme"


	def calc_residual(self, solHist, rhs, solver):
		
		# Account for cold start
		time_order = min(solver.iter, self.time_order)

		coeffs = self.coeffs[time_order-1]

		# Compute time derivative component
		residual = coeffs[0] * solHist[0]
		for iter_idx in range(1, time_order+1):
			residual += coeffs[iter_idx] * solHist[iter_idx]
		
		# Add RHS
		# NOTE: Negative convention here is for use with Newton's method
		residual = -(residual / self.dt) + rhs

		return residual
import numpy as np

from perform.rom.projectionROM.linearProjROM.linearProjROM import LinearProjROM


class LinearSPLSVTProj(LinearProjROM):
	"""
	Class for linear decoder and SP-LSVT formulation
	Trial basis is assumed to represent the conserved variables
	"""

	def __init__(self, model_idx, rom_domain, solver, sol_domain):

		if rom_domain.time_integrator.time_type == "explicit":
			raise ValueError("Explicit SP-LSVT not implemented yet")

		if ((rom_domain.time_integrator.time_type == "implicit")
				and (not rom_domain.time_integrator.dual_time)):
			raise ValueError("SP-LSVT is intended for primitive variable"
							+ "evolution, please use Galerkin or LSPG,"
							+ " or set dual_time = True")

		super().__init__(model_idx, rom_domain, solver, sol_domain)

	def calc_d_code(self, res_jacob, res, sol_domain):
		"""
		Compute change in low-dimensional state for implicit scheme
		Newton iteration
		"""

		# TODO: add hyper-reduction

		# TODO: scaled_trial_basis should be calculated once
		scaled_trial_basis = \
			self.trial_basis * self.norm_fac_prof_prim.ravel(order="C")[:, None]

		# compute test basis
		test_basis = (
			(res_jacob.toarray()
			/ self.norm_fac_prof_cons.ravel(order="C")[:, None])
			@ scaled_trial_basis
		)

		# lhs and rhs of Newton iteration
		lhs = test_basis.T @ test_basis
		rhs = (
			test_basis.T
			@ (res / self.norm_fac_prof_cons).ravel(order="C")
		)

		# linear solve
		dCode = np.linalg.solve(lhs, rhs)

		return dCode, lhs, rhs

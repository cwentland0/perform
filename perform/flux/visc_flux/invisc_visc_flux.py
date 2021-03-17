import numpy as np

from perform.constants import REAL_TYPE
from perform.flux.flux import Flux


class InviscViscFlux(Flux):
	"""
	Inviscid scheme, simply returns zero flux array
	"""

	def __init__(self, sol_domain, solver):

		super().__init__()

		self.flux = \
			np.zeros((sol_domain.gas_model.num_eqs, solver.mesh.num_cells + 1),
					dtype=REAL_TYPE)

	def calc_flux(self, sol_domain, solver):
		"""
		Just returns zero flux vector
		"""

		flux = self.flux
		return flux
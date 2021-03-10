from perform.timeIntegrator.explicitIntegrator import ClassicRK4, SSPRK3, JamesonLowStore
from perform.timeIntegrator.implicitIntegrator import BDF


def get_time_integrator(time_scheme, param_dict):
	"""
	Helper function to get time integrator object
	"""

	if (time_scheme == "bdf"):
		time_integrator = BDF(param_dict)
	elif (time_scheme == "classic_rk4"):
		time_integrator = ClassicRK4(param_dict)
	elif (time_scheme == "ssp_rk3"):
		time_integrator = SSPRK3(param_dict)
	elif (time_scheme == "jameson_low_store"):
		time_integrator = JamesonLowStore(param_dict)
	else:
		raise ValueError("Invalid choice of time_scheme: " + time_scheme)

	return time_integrator
from perform.time_integrator.explicit_integrator import ClassicRK4, SSPRK3, JamesonLowStore
from perform.time_integrator.implicit_integrator import BDF


def get_time_integrator(time_scheme, param_dict):
	"""
	Helper function to get time integrator object
	"""

	if (time_scheme == "BDF"):
		time_integrator = BDF(param_dict)
	elif (time_scheme == "ClassicRK4"):
		time_integrator = ClassicRK4(param_dict)
	elif (time_scheme == "SSPRK3"):
		time_integrator = SSPRK3(param_dict)
	elif (time_scheme == "JamesonLowStore"):
		time_integrator = JamesonLowStore(param_dict)
	else:
		raise ValueError("Invalid choice of time_scheme: " + time_scheme)

	return time_integrator
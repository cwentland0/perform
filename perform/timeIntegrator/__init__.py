from perform.timeIntegrator.explicitIntegrator import classicRK4, sspRK3, jamesonLowStore
from perform.timeIntegrator.implicitIntegrator import bdf


def getTimeIntegrator(timeScheme, paramDict):
	"""
	Helper function to get time integrator object
	"""

	if (timeScheme == "bdf"):
		timeIntegrator = bdf(paramDict)
	elif (timeScheme == "classicRK4"):
		timeIntegrator = classicRK4(paramDict)
	elif (timeScheme == "sspRK3"):
		timeIntegrator = sspRK3(paramDict)
	elif (timeScheme == "jamesonLowStore"):
		timeIntegrator = jamesonLowStore(paramDict)
	else:
		raise ValueError("Invalid choice of timeScheme: "+timeScheme)

	return timeIntegrator
from perform.rom.projectionROM.linearProjROM.linearGalerkinProj import linearGalerkinProj
from perform.rom.projectionROM.linearProjROM.linearLSPGProj import linearLSPGProj
from perform.rom.projectionROM.linearProjROM.linearSPLSVTProj import linearSPLSVTProj
from perform.rom.projectionROM.autoencoderProjROM.autoencoderGalerkinProjTFKeras import autoencoderGalerkinProjTFKeras


def getROMModel(modelIdx, romDomain, solver, solDomain):
	"""
	Helper function to retrieve various models
	Helps keep the clutter our of romDomain
	"""

	if (romDomain.romMethod == "linearGalerkinProj"):
		model = linearGalerkinProj(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "linearLSPGProj"):
		model = linearLSPGProj(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "linearSPLSVTProj"):
		model = linearSPLSVTProj(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "autoencoderGalerkinProjTFKeras"):
		model = autoencoderGalerkinProjTFKeras(modelIdx, romDomain, solver, solDomain)

	else:
		raise ValueError("Invalid ROM method name: "+romDomain.romMethod)

	return model


from perform.rom.projectionROM.linearProjROM.linearGalerkinProj import linearGalerkinProj
from perform.rom.projectionROM.autoencoderProjROM.autoencoderGalerkinProjTFKeras import autoencoderGalerkinProjTFKeras


def getROMModel(modelIdx, romDomain, solver, solDomain):

	if (romDomain.romMethod == "linearGalerkinProj"):
		model = linearGalerkinProj(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "autoencoderGalerkinProjTFKeras"):
		model = autoencoderGalerkinProjTFKeras(modelIdx, romDomain, solver, solDomain)

	else:
		raise ValueError("Invalid ROM method name: "+romDomain.romMethod)

	return model


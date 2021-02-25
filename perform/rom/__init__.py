# linear models
from perform.rom.projectionROM.linearProjROM.linearGalerkinProj import linearGalerkinProj
from perform.rom.projectionROM.linearProjROM.linearLSPGProj import linearLSPGProj
from perform.rom.projectionROM.linearProjROM.linearSPLSVTProj import linearSPLSVTProj

# TensorFlow-Keras autoencoder models
from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderGalerkinProjTFKeras import autoencoderGalerkinProjTFKeras
from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderLSPGProjTFKeras import autoencoderLSPGProjTFKeras
from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderSPLSVTProjTFKeras import autoencoderSPLSVTProjTFKeras

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

	elif (romDomain.romMethod == "autoencoderLSPGProjTFKeras"):
		model = autoencoderLSPGProjTFKeras(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "autoencoderSPLSVTProjTFKeras"):
		model = autoencoderSPLSVTProjTFKeras(modelIdx, romDomain, solver, solDomain)

	else:
		raise ValueError("Invalid ROM method name: "+romDomain.romMethod)

	return model


# linear models
from perform.rom.projectionROM.linearProjROM.linearGalerkinProj import linearGalerkinProj
from perform.rom.projectionROM.linearProjROM.linearLSPGProj import linearLSPGProj
from perform.rom.projectionROM.linearProjROM.linearSPLSVTProj import linearSPLSVTProj

# TensorFlow-Keras autoencoder models
TFKERAS_IMPORT_SUCCESS = True
try:
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # don't print all the TensorFlow warnings
	from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderGalerkinProjTFKeras import autoencoderGalerkinProjTFKeras
	from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderLSPGProjTFKeras import autoencoderLSPGProjTFKeras
	from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderSPLSVTProjTFKeras import autoencoderSPLSVTProjTFKeras

except ImportError:
	TFKERAS_IMPORT_SUCCESS = False
	

def getROMModel(modelIdx, romDomain, solver, solDomain):
	"""
	Helper function to retrieve various models
	Helps keep the clutter our of romDomain
	"""

	from perform.rom.projectionROM.autoencoderProjROM.autoencoderTFKeras.autoencoderSPLSVTProjTFKeras import autoencoderSPLSVTProjTFKeras

	# linear subspace methods
	if (romDomain.romMethod == "linearGalerkinProj"):
		model = linearGalerkinProj(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "linearLSPGProj"):
		model = linearLSPGProj(modelIdx, romDomain, solver, solDomain)

	elif (romDomain.romMethod == "linearSPLSVTProj"):
		model = linearSPLSVTProj(modelIdx, romDomain, solver, solDomain)

	# TensorFlow-Keras autoencoder models
	elif (romDomain.romMethod[-7:] == "TFKeras"):
		if TFKERAS_IMPORT_SUCCESS:

			if (romDomain.romMethod == "autoencoderGalerkinProjTFKeras"):
				model = autoencoderGalerkinProjTFKeras(modelIdx, romDomain, solver, solDomain)

			elif (romDomain.romMethod == "autoencoderLSPGProjTFKeras"):
				model = autoencoderLSPGProjTFKeras(modelIdx, romDomain, solver, solDomain)

			elif (romDomain.romMethod == "autoencoderSPLSVTProjTFKeras"):
				model = autoencoderSPLSVTProjTFKeras(modelIdx, romDomain, solver, solDomain)

			else:
				raise ValueError("Invalid TF-Keras ROM method name: " + romDomain.romMethod)

		else:
			raise ValueError("TF-Keras models failed to import, please check that TensorFlow >= 2.0 is installed")

	# TODO: PyTorch autoencoder models

	else:
		raise ValueError("Invalid ROM method name: "+romDomain.romMethod)

	return model


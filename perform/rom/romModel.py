from perform.constants import realType

import numpy as np
import pdb
import os

# TODO: workflow and method hierarchy can probably be cleaned up a lot

class romModel:
	"""
	Base class for any ROM model
	Assumed that every model may be equipped with some sort of data standardization requirement
	Also assumed that every model has some means of computing the full-dimensional state from the low
		dimensional state, i.e. a "decoder"
	"""

	def __init__(self, modelIdx, romDomain, solver):

		self.modelIdx 	= modelIdx
		self.latentDim 	= romDomain.latentDims[self.modelIdx]
		self.varIdxs 	= np.array(romDomain.modelVarIdxs[self.modelIdx], dtype=np.int32)
		self.numVars   	= len(self.varIdxs)
		self.solShape 	= (self.numVars, solver.mesh.numCells)

		# just copy some stuff for less clutter
		self.modelDir 	= romDomain.modelDir
		self.targetCons = romDomain.targetCons
		self.hyperReduc = romDomain.hyperReduc

		# low-dimensional state
		self.code = np.zeros(self.latentDim, dtype=realType)

		# get standardization profiles, if necessary
		self.normSubProfCons = None; self.normSubProfPrim = None
		self.normFacProfCons = None; self.normFacProfPrim = None
		self.centProfCons 	 = None; self.centProfPrim 	  = None
		if romDomain.hasConsNorm:
			self.normSubProfCons = self.loadStandardization(os.path.join(self.modelDir, romDomain.normSubConsIn[self.modelIdx]), default="zeros")
			self.normFacProfCons = self.loadStandardization(os.path.join(self.modelDir, romDomain.normFacConsIn[self.modelIdx]), default="ones")
		if romDomain.hasConsCent:
			self.centProfCons    = self.loadStandardization(os.path.join(self.modelDir, romDomain.centConsIn[self.modelIdx]), default="zeros")
		if romDomain.hasPrimNorm:
			self.normSubProfPrim = self.loadStandardization(os.path.join(self.modelDir, romDomain.normSubPrimIn[self.modelIdx]), default="zeros")
			self.normFacProfPrim = self.loadStandardization(os.path.join(self.modelDir, romDomain.normFacPrimIn[self.modelIdx]), default="ones")
		if romDomain.hasPrimCent:
			self.centProfPrim    = self.loadStandardization(os.path.join(self.modelDir, romDomain.centPrimIn[self.modelIdx]), default="zeros")


	def loadStandardization(self, standInput, default="zeros"):

		try:
			# TODO: add ability to accept single scalar value for standInput
			# 		catchList doesn't handle this when loading normSubIn, etc.
			
			# load single complete standardization profile from file
			standProf = np.load(standInput)
			assert (standProf.shape == self.solShape)
			return standProf

		except AssertionError:
			print("Standardization profile at " + standInput + " did not match solution shape")

		if (default == "zeros"):
			print("WARNING: standardization load failed or not specified, defaulting to zeros...")
			standProf = np.zeros(self.solShape, dtype=realType)
		elif (default == "ones"):
			print("WARNING: standardization load failed or not specified, defaulting to ones...")
			standProf = np.zeros(self.solShape, dtype=realType)

		return standProf


	def standardizeData(self, arr, 
						normalize=True, normFacProf=None, normSubProf=None,
						center=True, centProf=None,
						inverse=False):
		"""
		(de)centering and (de)normalization
		"""

		if normalize: 
			assert (normFacProf is not None), "Must provide normalization division factor to normalize"
			assert (normSubProf is not None), "Must provide normalization subtractive factor to normalize"
		if center:
			assert (centProf is not None), "Must provide centering profile to center"

		if inverse:
			if normalize: arr = self.normalize(arr, normFacProf, normSubProf, denormalize=True)
			if center: arr = self.center(arr, centProf, decenter=True)
		else:
			if center: arr = self.center(arr, centProf, decenter=False)
			if normalize: arr = self.normalize(arr, normFacProf, normSubProf, denormalize=False)
		
		return arr 


	def center(self, arr, centProf, decenter=False):
		"""
		(de)center input vector according to loaded centering profile
		"""

		if decenter:
			arr += centProf
		else:
			arr -= centProf
		return arr


	def normalize(self, arr, normFacProf, normSubProf, denormalize=False):
		"""
		(de)normalize input vector according to subtractive and divisive normalization profiles
		"""

		if denormalize:
			arr = arr * normFacProf + normSubProf
		else:
			arr = (arr - normSubProf) / normFacProf
		return arr


	def decodeSol(self, codeIn):
		"""
		Compute full decoding of solution, including decentering and denormalization
		"""

		sol = self.applyDecoder(codeIn)

		if (self.targetCons):
			sol = self.standardizeData(sol, 
								normalize=True, normFacProf=self.normFacProfCons, normSubProf=self.normSubProfCons,
								center=True, centProf=self.centProfCons, inverse=True)
		else:
			sol = self.standardizeData(sol, 
								normalize=True, normFacProf=self.normFacProfPrim, normSubProf=self.normSubProfPrim,
								center=True, centProf=self.centProfPrim, inverse=True)

		return sol


	def initFromCode(self, code0, solDomain):
		"""
		Initialize full-order solution from input low-dimensional state
		"""

		self.code = code0.copy()

		if (self.targetCons):
			solDomain.solInt.solCons[self.varIdxs, :] = self.decodeSol(self.code)
		else:
			solDomain.solInt.solPrim[self.varIdxs, :] = self.decodeSol(self.code)

		
	def updateSol(self, solDomain):
		"""
		Update solution after code has been updated
		"""

		# TODO: could just use this to replace initFromCode?

		if (self.targetCons):
			solDomain.solInt.solCons[self.varIdxs,:] = self.decodeSol(self.code)
		else:
			solDomain.solInt.solPrim[self.varIdxs, :] = self.decodeSol(self.code)
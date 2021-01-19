import pygems1d.constants as const

import re 
import numpy as np
import os
import pdb

def catchInput(inDict, inKey, defaultVal):
	"""
	Assign default values if user does not provide a certain input
	
	Inputs
	------
	inDict : dict
		Dictionary containing all input keys and values
	inKey : str
		Key to search inDict for
	default : varies
		Default value to assign if inKey is not found in inDict
		Also implicitly defines type to interpret value associated with inKey as

	Outputs
	-------
	outVal : varies
		Either the input value associated with inKey or 
	"""

	# TODO: correct error handling if default type is not recognized
	# TODO: check against lowercase'd strings so that inputs are not case sensitive. Do this for True/False too
	# TODO: instead of trusting user for NoneType, could also use NaN/Inf to indicate int/float defaults without passing a numerical default
	# 		or could just pass the actual default type lol, that'd be easier

	defaultType = type(defaultVal)
	try:
		# if NoneType passed as default, trust user
		if (defaultType == type(None)):
			outVal = inDict[inKey]
		else:
			outVal = defaultType(inDict[inKey])
	except:
		outVal = defaultVal

	return outVal


def catchList(inDict, inKey, default, lenHighest=1):
	"""
	Input processor for reading lists or lists of lists
	Default defines length of lists at lowest level
	"""

	# TODO: needs to throw an error if input list of lists is longer than lenHighest
	# TODO: could make a recursive function probably, just hard to define appropriate list lengths at each level

	listOfListsFlag = (type(default[0]) == list)
	
	try:
		inList = inDict[inKey]

		if (len(inList) == 0):
			raise ValueError

		# list of lists
		if listOfListsFlag:
			typeDefault = type(default[0][0])
			valList = []
			for listIdx in range(lenHighest):
				# if default type is NoneType, trust user
				if (typeDefault == type(None)):
					valList.append(inList[listIdx])
				else:
					castInList = [typeDefault(inVal) for inVal in inList[listIdx]]
					valList.append(castInList)

		# normal list
		else:
			typeDefault = type(default[0])
			# if default type is NoneType, trust user 
			if (typeDefault == type(None)):
				valList = inList
			else:
				valList = [typeDefault(inVal) for inVal in inList]

	except:
		if listOfListsFlag:
			valList = []
			for listIdx in range(lenHighest):
				valList.append(default[0])
		else:
			valList = default

	return valList


def parseValue(expr):
	"""
	Parse read text value into dict value
	"""

	try:
		return eval(expr)
	except:
		return eval(re.sub("\s+", ",", expr))
	else:
		return expr


def parseLine(line):
	"""
	Parse read text line into dict key and value
	"""

	eq = line.find('=')
	if eq == -1: raise Exception()
	key = line[:eq].strip()
	value = line[eq+1:-1].strip()
	return key, parseValue(value)


def readInputFile(inputFile):
	"""
	Read input file
	"""

	# TODO: better exception handling besides just a pass

	readDict = {}
	with open(inputFile) as f:
		contents = f.readlines()

	for line in contents: 
		try:
			key, val = parseLine(line)
			readDict[key] = val
			# convert lists to NumPy arrays
			if (type(val) == list): 
				readDict[key] = np.asarray(val)
		except:
			pass 

	return readDict


def parseBC(bcName, inDict):
	"""
	Parse boundary condition parameters from the input parameter dictionary
	"""

	# TODO: can definitely be made more general

	if ("press_"+bcName in inDict): 
		press = inDict["press_"+bcName]
	else:
		press = None 
	if ("vel_"+bcName in inDict): 
		vel = inDict["vel_"+bcName]
	else:
		vel = None 
	if ("temp_"+bcName in inDict):
		temp = inDict["temp_"+bcName]
	else:
		temp = None 
	if ("massFrac_"+bcName in inDict):
		massFrac = inDict["massFrac_"+bcName]
	else:
		massFrac = None
	if ("rho_"+bcName in inDict):
		rho = inDict["rho_"+bcName]
	else:
		rho = None
	if ("pertType_"+bcName in inDict):
		pertType = inDict["pertType_"+bcName]
	else:
		pertType = None
	if ("pertPerc_"+bcName in inDict):
		pertPerc = inDict["pertPerc_"+bcName]
	else:
		pertPerc = None
	if ("pertFreq_"+bcName in inDict):
		pertFreq = inDict["pertFreq_"+bcName]
	else:
		pertFreq = None
	
	return press, vel, temp, massFrac, rho, pertType, pertPerc, pertFreq


def getInitialConditions(solDomain, solver):
	"""
	Extract initial condition profile from two-zone initParamsFile, initFile .npy file, or restart file
	"""

	# TODO: add an option to interpolate a solution onto the given mesh, if different

	# intialize from restart file
	if solver.initFromRestart:
		solver.solTime, solPrim0, solver.restartIter = readRestartFile()

	# otherwise init from scratch IC or custom IC file 
	else:
		if (solver.initFile == None):
			solPrim0 = genPiecewiseUniformIC(solDomain, solver)
		else:
			# TODO: change this to .npz format with physical time included
			solPrim0 = np.load(solver.initFile)

	return solPrim0


def genPiecewiseUniformIC(solDomain, solver):
	"""
	Generate "left" and "right" states
	"""

	# TODO: generalize to >2 uniform regions

	if os.path.isfile(solver.icParamsFile):
		icDict 	= readInputFile(solver.icParamsFile)
	else:
		raise ValueError("Could not find initial conditions file at "+solver.icParamsFile)

	splitIdx 	= np.absolute(solver.mesh.xCell - icDict["xSplit"]).argmin()+1
	solPrim 	= np.zeros((solDomain.gasModel.numEqs, solver.mesh.numCells), dtype=const.realType)

	# left state
	solPrim[0,:splitIdx] 	= icDict["pressLeft"]
	solPrim[1,:splitIdx] 	= icDict["velLeft"]
	solPrim[2,:splitIdx] 	= icDict["tempLeft"]
	massFracLeft 			= icDict["massFracLeft"]
	assert(np.sum(massFracLeft) == 1.0), "massFracLeft must sum to 1.0"
	solPrim[3:,:splitIdx] 	= icDict["massFracLeft"][:-1]

	# right state
	solPrim[0,splitIdx:] 	= icDict["pressRight"]
	solPrim[1,splitIdx:] 	= icDict["velRight"]
	solPrim[2,splitIdx:] 	= icDict["tempRight"]
	massFracRight 			= icDict["massFracRight"]
	assert(np.sum(massFracRight) == 1.0), "massFracRight must sum to 1.0"
	solPrim[3:,splitIdx:] 	= massFracRight[:-1]
	
	return solPrim


def readRestartFile():
	"""
	Read solution state from restart file 
	"""

	# TODO: if higher-order multistep scheme, load previous time steps to preserve time accuracy

	# read text file for restart file iteration number
	with open(os.path.join(const.restartOutputDir, "restartIter.dat"), "r") as f:
		restartIter = int(f.read())

	# read solution
	restartFile = os.path.join(const.restartOutputDir, "restartFile_"+str(restartIter)+".npz")
	restartIn = np.load(restartFile)

	solTime = restartIn["solTime"].item() 	# convert array() to scalar
	solPrim = restartIn["solPrim"]

	restartIter += 1 # so this restart file doesn't get overwritten on next restart write

	return solTime, solPrim, restartIter
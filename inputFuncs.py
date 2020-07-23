import re 
import numpy as np
import pdb

# parse read text value into dict value
def parseValue(expr):
	try:
		return eval(expr)
	except:
		return eval(re.sub("\s+", ",", expr))
	else:
		return expr

# parse read text line into dict key and value
def parseLine(line):
	eq = line.find('=')
	if eq == -1: raise Exception()
	key = line[:eq].strip()
	value = line[eq+1:-1].strip()
	return key, parseValue(value)

# read input file
# TODO: better exception handling besides just a pass
def readInputFile(inputFile):

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

# parse boundary condition parameters from the input parameter dictionary
def parseBC(bcName, inDict):
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
    
    return press, vel, temp, massFrac, pertType, pertPerc, pertFreq
from math import sqrt, pow 

# static state
tempStat 	= 300.0
pressStat 	= 1000000.0
velStat 	= 10.0

# gas properties
Cp 	= 1538.22
R 	= 389.96247654784236

# calc stagnation properties
gamma 	= Cp / (Cp - R)
c 		= sqrt(gamma * R * tempStat) 
mach 	= velStat / c 

tempStag = tempStat * (1.0 + 0.5 * (gamma - 1.0) * mach**2)
pressStag = pressStat * pow(tempStag/tempStat, gamma/(gamma-1.0))

print("Stagnation temperature: "+str(tempStag))
print("Stagnation pressure: "+str(pressStag))
import numpy as np

from perform.constants import REAL_TYPE


# Constants to use throughout testing

# sample air chemistry dictionary
CHEM_DICT_AIR = {}
CHEM_DICT_AIR["num_species"] = 3
CHEM_DICT_AIR["mol_weights"] = np.array([32.0, 28.0, 40.0], dtype=REAL_TYPE)
CHEM_DICT_AIR["species_names"] = np.array(["oxygen", "nitrogen", "argon"])
CHEM_DICT_AIR["enth_ref"] = np.array([0.0, 0.0, 0.0], dtype=REAL_TYPE)
CHEM_DICT_AIR["cp"] = np.array([918.0, 1040.0, 520.3], dtype=REAL_TYPE)
CHEM_DICT_AIR["pr"] = np.array([0.730, 0.718, 0.687], dtype=REAL_TYPE)
CHEM_DICT_AIR["sc"] = np.array([0.612, 0.612, 0.612], dtype=REAL_TYPE)
CHEM_DICT_AIR["mu_ref"] = np.array([2.07e-5, 1.76e-5, 2.27e-5], dtype=REAL_TYPE)
CHEM_DICT_AIR["temp_ref"] = np.array([0.0, 0.0, 0.0], dtype=REAL_TYPE)

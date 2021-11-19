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

# sample reactant-product chemistry and reaction dictionaries
CHEM_DICT_REACT = {}
CHEM_DICT_REACT["num_species"] = 2
CHEM_DICT_REACT["mol_weights"] = np.array([21.32, 21.32], dtype=REAL_TYPE)
CHEM_DICT_REACT["species_names"] = np.array(["Reactant", "Product"])
CHEM_DICT_REACT["enth_ref"] = np.array([-7.4320e6, -10.8e6], dtype=REAL_TYPE)
CHEM_DICT_REACT["cp"] = np.array([1538.22, 1538.22], dtype=REAL_TYPE)
CHEM_DICT_REACT["pr"] = np.array([0.713, 0.713], dtype=REAL_TYPE)
CHEM_DICT_REACT["sc"] = np.array([0.62, 0.62], dtype=REAL_TYPE)
CHEM_DICT_REACT["mu_ref"] = np.array([7.35e-4, 7.35e-4], dtype=REAL_TYPE)
CHEM_DICT_REACT["temp_ref"] = np.array([0.0, 0.0], dtype=REAL_TYPE)

CHEM_DICT_REACT["num_reactions"] = 1
CHEM_DICT_REACT["nu"] = [[1.0, -1.0]]
CHEM_DICT_REACT["nu_arr"] = [[1.0, 0.0]]
CHEM_DICT_REACT["pre_exp_fact"] = [2.12e10]
CHEM_DICT_REACT["temp_exp"] = [0.0]
CHEM_DICT_REACT["act_energy"] = [2.025237e8]

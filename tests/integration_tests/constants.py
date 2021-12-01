import os
import shutil

import numpy as np

from perform.constants import REAL_TYPE, PARAM_INPUTS, ROM_INPUTS


# Constants to use throughout testing

# sample air chemistry dictionary
CHEM_DICT_AIR = {}
CHEM_DICT_AIR["gas_model"] = "cpg"
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
CHEM_DICT_REACT["gas_model"] = "cpg"
CHEM_DICT_REACT["num_species"] = 2
CHEM_DICT_REACT["mol_weights"] = np.array([21.32, 21.32], dtype=REAL_TYPE)
CHEM_DICT_REACT["species_names"] = np.array(["Reactant", "Product"])
CHEM_DICT_REACT["enth_ref"] = np.array([-7.4320e6, -10.8e6], dtype=REAL_TYPE)
CHEM_DICT_REACT["cp"] = np.array([1538.22, 1538.22], dtype=REAL_TYPE)
CHEM_DICT_REACT["pr"] = np.array([0.713, 0.713], dtype=REAL_TYPE)
CHEM_DICT_REACT["sc"] = np.array([0.62, 0.62], dtype=REAL_TYPE)
CHEM_DICT_REACT["mu_ref"] = np.array([7.35e-4, 7.35e-4], dtype=REAL_TYPE)
CHEM_DICT_REACT["temp_ref"] = np.array([0.0, 0.0], dtype=REAL_TYPE)

CHEM_DICT_REACT["reaction_model"] = "fr_irrev"
CHEM_DICT_REACT["num_reactions"] = 1
CHEM_DICT_REACT["nu"] = [[1.0, -1.0]]
CHEM_DICT_REACT["nu_arr"] = [[1.0, 0.0]]
CHEM_DICT_REACT["pre_exp_fact"] = [2.12e10]
CHEM_DICT_REACT["temp_exp"] = [0.0]
CHEM_DICT_REACT["act_energy"] = [2.025237e8]

# consistent primitive initial conditions
SOL_PRIM_IN_AIR = np.array(
    [
        [1e6, 1e5],
        [2.0, 1.0],
        [300.0, 400.0],
        [0.4, 0.6],
        [0.6, 0.4],
    ]
)
SOL_PRIM_IN_REACT = np.array(
    [
        [1e6, 9e5],
        [2.0, 1.0],
        [1000.0, 1200.0],
        [0.6, 0.4],
    ]
)

# generate directory which acts as PERFORM working directory
TEST_DIR = "test_dir"


def gen_test_dir():
    if os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.mkdir(TEST_DIR)


# delete working directory on cleanup
def del_test_dir():
    if os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)


# get output mode and directory
def get_output_mode():

    output_mode = bool(int(os.environ["PERFORM_TEST_OUTPUT_MODE"]))
    output_dir = os.environ["PERFORM_TEST_OUTPUT_DIR"]

    return output_mode, output_dir


# sample input files necessary for solution domain initialization
def solution_domain_setup(run_dir=TEST_DIR):

    # generate mesh file
    mesh_file = os.path.join(run_dir, "mesh.inp")
    with open(mesh_file, "w") as f:
        f.write("x_left = 0.0\n")
        f.write("x_right = 2e-5\n")
        f.write("num_cells = 2\n")

    # generate chemistry file
    chem_file = os.path.join(run_dir, "chem.inp")
    with open(chem_file, "w") as f:
        for key, item in CHEM_DICT_REACT.items():
            if isinstance(item, str):
                f.write(key + ' = "' + str(item) + '"\n')
            elif isinstance(item, list) or isinstance(item, np.ndarray):
                f.write(key + " = [" + ",".join(str(val) for val in item) + "]\n")
            else:
                f.write(key + " = " + str(item) + "\n")

    # generate solver input files
    inp_file = os.path.join(run_dir, PARAM_INPUTS)
    with open(inp_file, "w") as f:
        f.write("stdout = False \n")
        f.write('chem_file = "./chem.inp" \n')
        f.write('mesh_file = "./mesh.inp" \n')
        f.write('init_file = "test_init_file.npy" \n')
        f.write("dt = 1e-7 \n")
        f.write('time_scheme = "bdf" \n')
        f.write("adapt_dtau = True \n")
        f.write("time_order = 2 \n")
        f.write("num_steps = 10 \n")
        f.write("res_tol = 1e-11 \n")
        f.write('invisc_flux_scheme = "roe" \n')
        f.write('visc_flux_scheme = "standard" \n')
        f.write("space_order = 2 \n")
        f.write('grad_limiter = "barth_face" \n')
        f.write('bound_cond_inlet = "meanflow" \n')
        f.write("press_inlet = 1003706.0 \n")
        f.write("temp_inlet = 1000.0 \n")
        f.write("vel_inlet = 1853.0 \n")
        f.write("rho_inlet = 3944.0 \n")
        f.write("mass_fracs_inlet = [0.6, 0.4] \n")
        f.write('bound_cond_outlet = "meanflow" \n')
        f.write("press_outlet = 898477.0 \n")
        f.write("vel_outlet = 1522.0 \n")
        f.write("rho_outlet = 2958.0 \n")
        f.write("mass_fracs_outlet = [0.4, 0.6] \n")
        f.write("probe_locs = [-1.0, 5e-6, 1.0] \n")
        f.write('probe_vars = ["pressure", "temperature", "species-0", "density"] \n')
        f.write("save_restarts = True \n")
        f.write("restart_interval = 5 \n")
        f.write("out_interval = 2 \n")
        f.write("out_itmdt_interval = 5 \n")
        f.write("prim_out = True \n")
        f.write("cons_out = True \n")
        f.write("source_out = True \n")
        f.write("hr_out = True \n")
        f.write("rhs_out = True \n")
        f.write("vis_show = False \n")
        f.write("vis_save = True \n")
        f.write("vis_interval = 3 \n")
        f.write('vis_type_0 = "field" \n')
        f.write('vis_var_0 = ["temperature", "density", "pressure", "species-0"] \n')
        f.write("vis_y_bounds_0 = [[500, 1500], [1.8, 2.6], [1.2e6, 8e5], [-0.1, 1.1]] \n")
        f.write('vis_type_1 = "probe" \n')
        f.write('vis_var_1 = ["temperature", "density", "pressure", "species-0"] \n')
        f.write("vis_y_bounds_1 = [[500, 1500], [1.8, 2.6], [1.2e6, 8e5], [-0.1, 1.1]] \n")
        f.write("probe_num_1 = 1 \n")

    np.save(os.path.join(run_dir, "test_init_file.npy"), SOL_PRIM_IN_REACT)


def rom_domain_setup(run_dir=TEST_DIR, method="galerkin", space_mapping="linear", var_mapping="conservative"):

    assert method in ["galerkin", "lspg", "mplsvt"]
    assert space_mapping in ["linear", "nonlinear"]
    assert var_mapping in ["conservative", "primitive"]

    # presumably this has already been generated
    inp_file = os.path.join(run_dir, PARAM_INPUTS)
    with open(inp_file, "a") as f:
        f.write("calc_rom = True \n")

    # generate ROM input file
    inp_file = os.path.join(run_dir, ROM_INPUTS)
    model_dir = os.path.join(run_dir, "model_files")
    os.mkdir(model_dir)
    with open(inp_file, "w") as f:
        f.write('rom_method = "' + method + '" \n')
        f.write('var_mapping = "' + var_mapping + '" \n')
        f.write('space_mapping = "' + space_mapping + '" \n')
        f.write("num_models = 1 \n")
        f.write("latent_dims = [8] \n")
        f.write("model_var_idxs = [[0, 1, 2, 3]] \n")
        f.write('model_dir = "' + model_dir + '" \n')
        f.write('basis_files = ["spatial_modes.npy"] \n')
        f.write('cent_prim = ["cent_prof_prim.npy"] \n')
        f.write('cent_cons = ["cent_prof_cons.npy"] \n')
        f.write('norm_sub_prim = ["norm_sub_prof_prim.npy"] \n')
        f.write('norm_fac_prim = ["norm_fac_prof_prim.npy"] \n')
        f.write('norm_sub_cons = ["norm_sub_prof_cons.npy"] \n')
        f.write('norm_fac_cons = ["norm_fac_prof_cons.npy"] \n')

    # generate model files, standardization profiles
    modes = np.reshape(np.eye(8), (4, 2, 8))
    cent_prof = np.zeros((4, 2), dtype=REAL_TYPE)
    norm_sub_prof = np.zeros((4, 2), dtype=REAL_TYPE)
    norm_fac_prof = np.ones((4, 2), dtype=REAL_TYPE)

    np.save(os.path.join(model_dir, "spatial_modes.npy"), modes)
    np.save(os.path.join(model_dir, "cent_prof_prim.npy"), cent_prof)
    np.save(os.path.join(model_dir, "cent_prof_cons.npy"), cent_prof)
    np.save(os.path.join(model_dir, "norm_sub_prof_prim.npy"), norm_sub_prof)
    np.save(os.path.join(model_dir, "norm_fac_prof_prim.npy"), norm_fac_prof)
    np.save(os.path.join(model_dir, "norm_sub_prof_cons.npy"), norm_sub_prof)
    np.save(os.path.join(model_dir, "norm_fac_prof_cons.npy"), norm_fac_prof)

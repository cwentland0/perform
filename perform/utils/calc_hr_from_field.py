from math import ceil

import numpy as np

from perform.input_funcs import read_input_file
from perform.solution.solution_phys import SolutionPhys
from perform.gas_model.calorically_perfect_gas import CaloricallyPerfectGas
from perform.reaction_model.finite_rate_irrev_reaction import FiniteRateIrrevReaction


def calc_hr_from_field(chem_file, gas_model_str, reac_model_str, dt, prim_file=None, cons_file=None, num_prints=10):
    """Computes heat release rate from provided primitive or conservative field data array."""

    # load chem dict
    chem_dict = read_input_file(chem_file)

    # load gas model
    if gas_model_str == "cpg":
        gas_model = CaloricallyPerfectGas(chem_dict)
    else:
        raise ValueError("Invalid gas_model_str: " + str(gas_model_str))

    # load reaction model
    if reac_model_str == "fr_irrev":
        reaction_model = FiniteRateIrrevReaction(gas_model, chem_dict)
    else:
        raise ValueError("Invalid reac_model_str: " + str(reac_model_str))

    # load prim/cons file, assumed to be in [num_vars, num_cells, num_snaps] format
    assert (prim_file is not None) != (
        cons_file is not None
    ), "Can only provide one of prim_file or cons_file, not both"
    if prim_file is not None:
        prim = True
        prim_data = np.load(prim_file)
        dtype = prim_data.dtype
        num_cells, num_snaps = prim_data.shape[1:]
        sol = SolutionPhys(gas_model, num_cells, sol_prim_in=prim_data[:, :, 0])
    else:
        prim = False
        cons_data = np.load(cons_file)
        dtype = cons_data.dtype
        num_cells, num_snaps = cons_data.shape[1:]
        sol = SolutionPhys(gas_model, num_cells, sol_cons_in=cons_data[:, :, 0])

    hr_data = np.zeros((num_cells, num_snaps), dtype=dtype)

    # loop over data, calculate heat release at each step
    print_iter = ceil(num_snaps / num_prints)
    for idx in range(num_snaps):

        if (idx % print_iter) == 0:
            print("Processing snapshot " + str(idx + 1) + "/" + str(num_snaps))

        if prim:
            sol.sol_prim = prim_data[:, :, idx]
        else:
            sol.sol_cons = cons_data[:, :, idx]
        sol.update_state(prim)

        _, _, hr_data[:, idx] = reaction_model.calc_reaction(sol, dt)

    return hr_data


if __name__ == "__main__":

    while True:
        print("Enter number of desired gas model")
        print("1: calorically-perfect gas")
        gas_model_input = input()
        if gas_model_input == "1":
            gas_model_str = "cpg"
            break

    while True:
        print("Enter number of desired reaction model")
        print("1: finite-rate irreversible")
        reac_model_input = input()
        if reac_model_input == "1":
            reac_model_str = "fr_irrev"
            break

    chem_file = input("Enter chemistry file path: ")

    while True:
        print("Enter field type number")
        print("1: primitive field")
        print("2: conservative field")
        field_type_input = input()
        if field_type_input == "1":
            prim = True
            break
        elif field_type_input == "2":
            prim = False
            break

    dt = float(input("Enter time step, in seconds: "))

    # TODO: get output directory, or use same directory as prim/cons field?

    if prim:
        prim_file = input("Enter primitive file path: ")
        hr_data = calc_hr_from_field(chem_file, gas_model_str, reac_model_str, dt, prim_file=prim_file)
    else:
        cons_file = input("Enter conservative file path: ")
        hr_data = calc_hr_from_field(chem_file, gas_model_str, reac_model_str, dt, cons_file=cons_file)

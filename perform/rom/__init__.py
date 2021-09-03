from packaging import version
from time import sleep

# # linear models
# from perform.rom.projection_rom.linear_proj_rom.linear_galerkin_proj import LinearGalerkinProj
# from perform.rom.projection_rom.linear_proj_rom.linear_lspg_proj import LinearLSPGProj
# from perform.rom.projection_rom.linear_proj_rom.linear_splsvt_proj import LinearSPLSVTProj

# Check whether ML libraries are accessible
# Tensorflow-Keras
# TFKERAS_IMPORT_SUCCESS = True
# try:
#     import os

#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # don't print all the TensorFlow warnings
#     import tensorflow as tf
#     from perform.rom.ml_library.tfkeras_library import TFKerasLibrary

# except ImportError:
#     TFKERAS_IMPORT_SUCCESS = False

# # PyTorch
# TORCH_IMPORT_SUCCESS = True
# try:
#     import torch
#     from perform.rom.ml_library.pytorch_library import PyTorchLibrary

# except ImportError:
#     TORCH_IMPORT_SUCCESS = False

# # ML models
# if TFKERAS_IMPORT_SUCCESS or TORCH_IMPORT_SUCCESS:
#     from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_galerkin_proj import AutoencoderGalerkinProj
#     from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_lspg_proj import AutoencoderLSPGProj
#     from perform.rom.projection_rom.autoencoder_proj_rom.autoencoder_splsvt_proj import AutoencoderSPLSVTProj

# # list of valid model strings for each ML library
# tf_keras_classes = ["autoencoder_galerkin_proj", "autoencoder_lspg_proj", "autoencoder_splsvt_proj"]
# pytorch_classes = []
# ml_classes = list(set(tf_keras_classes + pytorch_classes))

# ROM methods
from perform.rom.rom_method.projection_method.galerkin_projection import GalerkinProjection
from perform.rom.rom_method.projection_method.lspg_projection import LSPGProjection

# variable mappings
from perform.rom.rom_variable_mapping.prim_variable_mapping import PrimVariableMapping
from perform.rom.rom_variable_mapping.cons_variable_mapping import ConsVariableMapping
from perform.rom.rom_variable_mapping.lifted_xi_c_variable_mapping import LiftedXiCVariableMapping

# time steppers
from perform.rom.rom_time_stepper.numerical_stepper import NumericalStepper

# space mappings
from perform.rom.rom_space_mapping.linear_space_mapping import LinearSpaceMapping

def get_rom_method(rom_method, sol_domain, rom_domain):

    if rom_method == "galerkin":
        return GalerkinProjection(sol_domain, rom_domain)
    elif rom_method == "lspg":
        return LSPGProjection(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM rom_method: " + str(rom_method))


def get_variable_mapping(var_mapping, sol_domain, rom_domain):

    if var_mapping == "primitive":
        return PrimVariableMapping(sol_domain, rom_domain)
    elif var_mapping == "conservative":
        return ConsVariableMapping(sol_domain, rom_domain)
    elif var_mapping == "lifted-xi-c":
        return LiftedXiCVariableMapping(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM var_mapping: " + str(var_mapping))


def get_time_stepper(time_stepper, sol_domain, rom_domain):

    if time_stepper == "numerical":
        return NumericalStepper(sol_domain, rom_domain)
    else:
        raise ValueError("Something went wrong, rom_method set invalid time_stepper: " + str(time_stepper))


def get_space_mapping(space_mapping, sol_domain, rom_domain):

    if space_mapping == "linear":
        return LinearSpaceMapping(sol_domain, rom_domain)
    else:
        raise ValueError("Invalid ROM space_mapping: " + str(space_mapping))


# def get_rom_model(model_idx, rom_domain, sol_domain):
#     """Helper function to retrieve ROM models"""

#     # linear subspace projection methods
#     if rom_domain.rom_method == "linear_galerkin_proj":
#         model = LinearGalerkinProj(model_idx, rom_domain, sol_domain)

#     elif rom_domain.rom_method == "linear_lspg_proj":
#         model = LinearLSPGProj(model_idx, rom_domain, sol_domain)

#     elif rom_domain.rom_method == "linear_splsvt_proj":
#         model = LinearSPLSVTProj(model_idx, rom_domain, sol_domain)

#     # ML models
#     elif rom_domain.rom_method in ml_classes:

#         # check that desired ML library is installed and requested model is compatible
#         rom_domain.mllib_name = rom_domain.rom_dict["mllib_name"]
#         if rom_domain.mllib_name == "tfkeras":
#             assert TFKERAS_IMPORT_SUCCESS, "Tensorflow failed to import, please check that it is installed"
#             assert rom_domain.rom_method in tf_keras_classes, (
#                 "Requested ROM method (" + rom_domain.rom_method + ") is not available with Tensorflow."
#             )
#             if version.parse(tf.__version__) < version.parse("2.4"):
#                 print("WARNING: You are using TensorFlow version < 2.4, proper ROM behavior not guaranteed")
#                 sleep(1.0)
#             rom_domain.mllib = TFKerasLibrary(rom_domain)

#         elif rom_domain.mllib_name == "pytorch":
#             assert TORCH_IMPORT_SUCCESS, "PyTorch failed to import, please check that it is installed."
#             assert rom_domain.rom_method in pytorch_classes, (
#                 "Requested ROM method (" + rom_domain.rom_method + ") is not available with PyTorch."
#             )
#             rom_domain.mllib = PyTorchLibrary(rom_domain)

#         else:
#             raise ValueError("Invalid mllib_name: " + str(rom_domain.mllib_name))

#         # initialize ML models, finally
#         # non-linear manifold projection via autoencoder methods
#         if rom_domain.rom_method == "autoencoder_galerkin_proj":
#             model = AutoencoderGalerkinProj(model_idx, rom_domain, sol_domain)

#         elif rom_domain.rom_method == "autoencoder_lspg_proj":
#             model = AutoencoderLSPGProj(model_idx, rom_domain, sol_domain)

#         elif rom_domain.rom_method == "autoencoder_splsvt_proj":
#             model = AutoencoderSPLSVTProj(model_idx, rom_domain, sol_domain)

#     else:
#         raise ValueError("Invalid ROM method name: " + rom_domain.rom_method)

#     return model

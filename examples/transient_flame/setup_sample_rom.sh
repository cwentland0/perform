#!/bin/bash

# check that current dir at least has inputs directory
if [ ! -d "inputs" ]; then
	echo "The inputs/ directory does not exist in this directory, are you executing this script in the standing_flame/ directory?"
	exit 1
fi

# check if user is okay with overwrite
read -p "Retrieving the sample ROM files will overwrite the solver_params.inp file, are you okay with this? (y/n): " overwrite_flag
if [ "${overwrite_flag}" = "y" ]; then
	:
elif [ "${overwrite_flag}" = "n" ]; then
	echo "Aborting..."
	exit 2
else
	echo "Invalid user input, aborting..."
	exit 3
fi

# retrieve and unpack ROM files
echo "Retrieving sample files..."
wget -q --show-progress --no-check-certificate 'https://docs.google.com/uc?export=download&id=11MfQVGaW9U4U3rQT2OM9eardJ9Lk7507' -O transient_flame_autoencoder_splsvt_proj_tfkeras_rom_sample.zip

echo "Unpacking sample files..."
unzip -q transient_flame_autoencoder_splsvt_proj_tfkeras_rom_sample.zip
mv sol_prim_init_100mus.npy ./inputs

# set path to model_dir
sed -i "9s#.*#model_dir      = \"${PWD}/sample_cae_data\"#" rom_params.inp

rm transient_flame_autoencoder_splsvt_proj_tfkeras_rom_sample.zip

echo "Sample ROM set up. Execute \"perform .\" to test it!"

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
wget -q --show-progress --no-check-certificate 'https://docs.google.com/uc?export=download&id=1T5LlPZJNkGd8iAoNwb99H2RvTzANRyrr' -O standing_flame_linear_splsvt_proj_rom_sample.zip

echo "Unpacking sample files..."
unzip -q standing_flame_linear_splsvt_proj_rom_sample.zip
mv sol_prim_init_20mus.npy ./inputs

# set path to model_dir
sed -i "9s#.*#model_dir      = \"${PWD}/sample_pod_data\"#" rom_params.inp

rm standing_flame_linear_splsvt_proj_rom_sample.zip

echo "Sample ROM set up. Execute \"perform .\" to test it!"

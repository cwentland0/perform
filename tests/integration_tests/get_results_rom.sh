#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1R1UIiq2HNDyfQ-fznHAzd9TEk42gasZG"
file_name="integration_results_rom.zip"

output_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/output_dir_rom"
file_path=${output_dir}/${file_name}

echo "Retrieving integration test truth results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${output_dir}
rm -f ${file_path}
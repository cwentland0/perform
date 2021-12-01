#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1xdCqGrPtF8kx3caWMkwIiydzuEM8wtjI"
file_name="integration_results_driver.zip"

output_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/output_dir_driver"
file_path=${output_dir}/${file_name}

echo "Retrieving integration test truth results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${output_dir}
rm -f ${file_path}

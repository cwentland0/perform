#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1y7N44dg6SId6hBWb2fK1-VND9R-S66oc"
file_name="standing_flame_results.zip"

local_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
file_path=${local_dir}/${file_name}

echo "Retrieving standing flame FOM results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${local_dir}
rm -f ${file_path}

#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1ucE9Pan86E41ZX_JIc5Sp94b0sVxS9oQ"
file_name="shock_tube_results.zip"

local_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
file_path=${local_dir}/${file_name}

echo "Retrieving shock tube FOM results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${local_dir}
rm -f ${file_path}

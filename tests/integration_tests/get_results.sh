#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1JJNQ3SWrm5BgILh7EbMAagpvO00RCzv2"
file_name="integration_results.zip"

output_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/output_dir"
file_path=${output_dir}/${file_name}

echo "Retrieving integration test truth results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${output_dir}
rm -f ${file_path}

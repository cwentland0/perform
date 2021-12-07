#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1lUoz8gasiqjA9VkZzyer7Sl7yGn1lhSG"
file_name="contact_surface_results.zip"

local_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
file_path=${local_dir}/${file_name}

echo "Retrieving contact surface FOM results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${local_dir}
rm -f ${file_path}

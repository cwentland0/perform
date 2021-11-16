#!/bin/bash

# Retrieves results from remote storage

gdrive_id="1QPQTH0XTTFZqLPScNU4-LN5IZagoYBn1"
file_name="transient_flame_results.zip"

local_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
file_path=${local_dir}/${file_name}

echo "Retrieving transient flame FOM results..."
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
gdown -q --no-check-certificate -O ${file_path} --id ${gdrive_id}
unzip -o -q ${file_path} -d ${local_dir}
rm -f ${file_path}

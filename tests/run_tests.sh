#!/bin/bash

testdir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Run basic tests
echo "Running unit tests..."
python3 ${testdir}/unit_tests/test_suite.py
unit_code=$?

echo "Running FOM integration tests..."
python3 ${testdir}/integration_tests/test_suite.py
fom_int_code=$?

echo "Running driver test..."
python3 ${testdir}/integration_tests/test_driver.py
driver_code=$?

echo "Running ROM integration tests..."
python3 ${testdir}/integration_tests/test_suite_rom.py
rom_int_code=$?

# Print final test stats
if [ ${unit_code} == 0 ]; then
    echo "Unit tests SUCCEEDED!"
else
    echo "Unit tests FAILED!"
fi
if [ ${fom_int_code} == 0 ]; then
    echo "FOM integration tests SUCCEEDED!"
else
    echo "FOM integration tests FAILED!"
fi
if [ ${driver_code} == 0 ]; then
    echo "Driver tests SUCCEEDED!"
else
    echo "Driver tests FAILED!"
fi
if [ ${rom_int_code} == 0 ]; then
    echo "ROM integration tests SUCCEEDED!"
else
    echo "ROM integration tests FAILED!"
fi

# Run regression tests, if desired
while : ; do
    echo "Run regression tests? (y/n) "
    read run_reg
    if [[ ${run_reg} == "y" || ${run_reg} == "n" ]]; then
        break
    fi
done

if [ ${run_reg} == "y" ]; then
    echo "Running regression tests..."
    python3 ${testdir}/regression_tests/test_suite.py
    reg_code=$?
    if [ ${reg_code} == 0 ]; then
        echo "Regression tests SUCCEEDED!"
    else
        echo "Regression tests FAILED!"
    fi
else
    echo "Not running regression tests..."
fi


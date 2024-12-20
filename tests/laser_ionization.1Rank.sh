#! /usr/bin/env bash

#
# This file is part of HiPACE++.
#
# Authors: EyaDammak


# This file is part of the HiPACE++ test suite.
# It runs a Hipace simulation with in neutral hydrogen that gets ionized by the laser

# abort on first encounted error
set -eu -o pipefail

# Read input parameters
HIPACE_EXECUTABLE=$1
HIPACE_SOURCE_DIR=$2

HIPACE_EXAMPLE_DIR=${HIPACE_SOURCE_DIR}/examples/laser_ionization
HIPACE_TEST_DIR=${HIPACE_SOURCE_DIR}/tests

FILE_NAME=`basename "$0"`
TEST_NAME="${FILE_NAME%.*}"

rm -rf $TEST_NAME

# Run the simulation
mpiexec -n 1 $HIPACE_EXECUTABLE $HIPACE_EXAMPLE_DIR/input_laser_ionization_linear

# Compare the result with theory
$HIPACE_EXAMPLE_DIR/analysis_laser_ionization.py

# Compare the results with checksum benchmark
$HIPACE_TEST_DIR/checksum/checksumAPI.py \
    --evaluate \
    --file_name $TEST_NAME \
    --test-name $TEST_NAME


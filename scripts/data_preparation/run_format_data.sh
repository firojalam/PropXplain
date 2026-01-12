#!/bin/bash

# This script is an example of how to format the raw data into the format required for training.
# You will need to adapt the python script and its arguments to your specific data.

# Activate your conda environment
# conda activate propxplain

# Set your paths
INPUT_DIR="original_data/arabic"
OUTPUT_DIR="data/arabic"
PYTHON_SCRIPT="bin/data_processing/convert_to_json.py"

echo "Formatting data for Arabic..."

python $PYTHON_SCRIPT \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --lang arabic

echo "Data formatting complete."

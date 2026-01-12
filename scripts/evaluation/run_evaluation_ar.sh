#!/bin/bash

# This script is an example of how to evaluate the model's predictions.
# You will need to adapt the python script and its arguments to your specific setup.

# Activate your conda environment
# conda activate propxplain

# Set your paths
PREDICTIONS_FILE="results/predictions_ar.jsonl"
GOLD_FILE="data/arabic/test.jsonl"
PYTHON_SCRIPT="bin/text-classification/exp_utils.py" # Assuming you have a script for evaluation

echo "Evaluating predictions for Arabic..."

python $PYTHON_SCRIPT \
    --predictions_file $PREDICTIONS_FILE \
    --gold_file $GOLD_FILE

echo "Evaluation complete."

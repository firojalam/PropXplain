#!/bin/bash

# This script is an example of how to run inference with a fine-tuned model.
# You will need to adapt the python script and its arguments to your specific setup.

# Activate your conda environment
# conda activate propxplain

# Set your paths
MODEL_PATH="models_output/llama3.1-8b-arabic-finetuned"
TEST_FILE="data/arabic/test.jsonl"
OUTPUT_FILE="results/predictions_ar.jsonl"
PYTHON_SCRIPT="bin/text-classification/run_inference.py"

echo "Running inference for Arabic..."

python $PYTHON_SCRIPT \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $TEST_FILE \
    --output_file $OUTPUT_FILE

echo "Inference complete."

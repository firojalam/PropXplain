#!/bin/bash

# This script is an example of how to run training for the Arabic model.
# You will need to adapt the python script and its arguments to your specific setup.

# Activate your conda environment
# conda activate propxplain

# Set your paths and parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TRAIN_FILE="data/arabic/train.jsonl"
DEV_FILE="data/arabic/dev.jsonl"
OUTPUT_DIR="models_output/llama3.1-8b-arabic-finetuned"
PYTHON_SCRIPT="bin/text-classification/llm_training.py"

echo "Starting training for Arabic..."

python $PYTHON_SCRIPT \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 24 \
    --save_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --load_best_model_at_end True

echo "Training complete."

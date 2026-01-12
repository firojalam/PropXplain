import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------------------
# Configurations
# -------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Replace with actual Llama 3.2 HF path if available
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
NUM_LABELS = 5
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
OUTPUT_DIR = "./llama3_label_finetune"
SEED = 42

USE_QLORA = True   # True: QLoRA (4bit); False: LoRA (fp16/bf16)

# -------------------------------
# Dataset
# -------------------------------

class LabelDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

# -------------------------------
# Metrics
# -------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# -------------------------------
# Main
# -------------------------------

def main():
    set_seed(SEED)

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Fixes pad_token error for Llama
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load Model ---
    if USE_QLORA:
        # QLoRA: load 4bit, prepare for QLoRA
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            },
            ignore_mismatched_sizes=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # LoRA (no quantization)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            ignore_mismatched_sizes=True
        )

    # --- Apply LoRA/QLoRA Adapter ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    # --- Prepare Data ---
    train_dataset = LabelDataset(TRAIN_FILE, tokenizer)
    val_dataset = LabelDataset(VAL_FILE, tokenizer)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        save_total_limit=2,
        fp16=True,   # Set to False if not using GPU/AMP
        report_to="none"
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # --- Train ---
    trainer.train()

    # --- Save Adapter ---
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    tokenizer.save_pretrained(OUTPUT_DIR)

    # --- Evaluate ---
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)


if __name__ == "__main__":
    main()
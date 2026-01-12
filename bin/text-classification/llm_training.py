import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_utils import set_seed
import numpy as np
import gc
import argparse

import json
import os
import gc
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from transformers import BitsAndBytesConfig

class LabelDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512, label_map=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [json.loads(line) for line in open(filename, encoding='utf-8')]
        self.label_map = label_map or {"false": 0, "true": 1}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item.get("text", item.get("paragraph"))
        label = self.label_map[item["label"]]
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def main(args):
    set_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config if using QLoRA
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        quantization_config=bnb if args.use_lora else None,
        torch_dtype=torch.float16 if args.use_qlora else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA config
    if args.use_lora:
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, peft_cfg)

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # Freeze/unfreeze all or Freeze then unfreeze last block and classifier
    # for name, param in model.named_parameters():
        # param.requires_grad = False
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point or param.dtype.is_complex:
            param.requires_grad = True
            
    # unfreeze classifier head
    for param in model.score.parameters():
        param.requires_grad = True

    # optionally unfreeze last transformer block (float parts)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            last = module[-1]
            for p in last.parameters():
                if p.dtype.is_floating_point:
                    p.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print("Trainable %d/%d params" % (
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters())
    ))

    train_ds = LabelDataset(args.train_file, tokenizer, max_length=args.max_len)
    eval_ds = LabelDataset(args.val_file, tokenizer, max_length=args.max_len)
    data_collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_file,
        per_device_eval_batch_size=args.val_file,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.use_qlora,
        optim="paged_adamw_32bit",
        logging_dir="./logs",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    cleanup_cuda()
    print("Eval:", trainer.evaluate())


def get_args():
    parser = argparse.ArgumentParser(description="Llama-3.1-8B-Instruct Fine-Tuning Configuration")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Hugging Face model path or ID")
    parser.add_argument("--train_file", type=str,
                        default="./original_data/arabic/ArProBinary/ArMPro_binary_train.jsonl",
                        help="Path to the training JSONL file")
    parser.add_argument("--val_file", type=str,
                        default="./original_data/arabic/ArProBinary/ArMPro_binary_dev.jsonl",
                        help="Path to the validation JSONL file")

    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of output labels/classes")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Single batch size (legacy, not used if using train/eval batch)")
    parser.add_argument("--train_batch", type=int, default=8,
                        help="Train batch size per device")
    parser.add_argument("--eval_batch", type=int, default=8,
                        help="Eval batch size per device")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str,
                        default="./exp/Llama-3.1-8B-Instruct/",
                        help="Directory for experiment outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Enable LoRA parameter-efficient tuning")
    parser.add_argument("--use_qlora", action="store_true", default=False,
                        help="Enable QLoRA 4-bit parameter-efficient tuning")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Maximum sequence length for inputs")

    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    main(args)
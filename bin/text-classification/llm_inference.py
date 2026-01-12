from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed
)
import torch
import json
import numpy as np
import os
import gc
import argparse
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import logging
import sys


logging.basicConfig(level=logging.INFO)
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
        
def inference_and_metrics(model_dir, test_file, batch_size=8, max_length=512):
    # load
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # build dataset
    test_dataset = LabelDataset(test_file, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(**inputs)            
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_mi, recall_mi, f1_mi, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    
    accuracy = accuracy_score(y_true, y_pred)

    print(f"🔹 Accuracy:  {accuracy:.4f}")
    # print(f"🔹 Precision: {precision:.4f}")
    # print(f"🔹 Recall:    {recall:.4f}")
    print(f"🔹 F1‑Score (macro):  {f1_ma:.4f}")
    print(f"🔹 F1‑Score (micro):  {f1_mi:.4f}")

    return {
        "accuracy": accuracy,
        "micro-f1": f1_mi,
        "macro-f1": f1_ma
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Inference and Metrics")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    results = inference_and_metrics(
        model_dir=args.model_dir,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    logging.info("Results: %s", results)
    


if __name__ == "__main__":
    main()
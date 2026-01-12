#!/usr/bin/env python3
import argparse
import json
import os
from typing import List

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

from swift.llm import PtEngine, RequestConfig, InferRequest, safe_snapshot_download, get_model_tokenizer, get_template
from swift.tuners import Swift

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA-tuned fake news classifier with SWIFT")
    parser.add_argument("--model", type=str, required=True, help="Base model ID (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--adapters", type=str, required=True, help="Path or HF repo for LoRA adapters")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Test dataset in JSONL with 'text' and 'label'")
    parser.add_argument("--output_preds", type=str, default="predictions.jsonl", help="File to save predictions")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=1, help="Limit generation length (binary output)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for inference")
    return parser.parse_args()

def load_test_examples(path: str) -> List[dict]:
    with open(path, "r", encoding="utf‑8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_inference(model_id, adapters, examples, batch_size, max_new_tokens, temperature):
    model, tokenizer = get_model_tokenizer(model_id)
    model = Swift.from_pretrained(model, safe_snapshot_download(adapters))
    template = get_template(model.model_meta.template, tokenizer)
    engine = PtEngine.from_model_template(model, template, max_batch_size=batch_size)
    cfg = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    preds = []
    for ex in examples:        
        req = InferRequest(
            messages=[
                {"role": "system", "content": ex["system_instruction"]},
                {
                    "role": "user",
                    "content": (
                        "Provide label and explanation. "
                        + ex["user_instruction"]
                        + "\n"
                        + ex["news"]
                        + "\nlabel:\n\nexplanation:"
                    ),
                },
            ]
        )        
        resp = engine.infer([req], cfg)[0]
        predicted = resp.choices[0].message.content.strip().lower()
        preds.append(predicted)
    return preds

def compute_and_log_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nFull classification report:\n", report)

def save_predictions(examples, y_pred, output_path):
    out = []
    for ex, pred in zip(examples, y_pred):
        rec = {**ex, "predicted": pred}
        out.append(rec)
    # Save as JSONL
    with open(output_path, "w", encoding="utf‑8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(out)} predictions to {output_path}")

def main():
    args = parse_args()
    examples = load_test_examples(args.test_jsonl)
    y_true = [ex["label"].lower() for ex in examples]
    y_pred = run_inference(
        args.model, args.adapters, examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    compute_and_log_metrics(y_true, y_pred)
    save_predictions(examples, y_pred, args.output_preds)

if __name__ == "__main__":
    main()

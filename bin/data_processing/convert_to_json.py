#!/usr/bin/env python3
"""
Convert raw data to JSONL format for PropXplain.

This script converts raw propaganda detection data into the standardized
JSONL format used for training and evaluation.

Usage:
    python convert_to_json.py --input_file <input> --output_file <output>
"""

import json
import argparse
from pathlib import Path


def convert_to_jsonl(input_file: str, output_file: str, label_key: str = "label"):
    """
    Convert raw data to JSONL format.
    
    Args:
        input_file: Path to input file
        output_file: Path to output JSONL file
        label_key: Key name for label field
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Parse input line (adjust based on your format)
            try:
                data = json.loads(line)
                
                # Standardize format
                output_obj = {
                    "input": data.get("text", data.get("input", "")),
                    "label": data.get(label_key, data.get("label", "")),
                }
                
                # Include explanation if present
                if "explanation" in data:
                    output_obj["explanation"] = data["explanation"]
                
                outfile.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                continue


def main():
    parser = argparse.ArgumentParser(description="Convert raw data to JSONL format")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--label_key", type=str, default="label",
                        help="Key name for label field in input data")
    
    args = parser.parse_args()
    
    print(f"Converting {args.input_file} to {args.output_file}...")
    convert_to_jsonl(args.input_file, args.output_file, args.label_key)
    print("Conversion complete!")


if __name__ == "__main__":
    main()

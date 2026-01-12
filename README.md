# PropXplain: Can LLMs Enable Explainable Propaganda Detection?

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-EMNLP_2025-red.svg)](https://aclanthology.org/2025.findings-emnlp.1296/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PropXplain-yellow.svg)](https://huggingface.co/datasets/QCRI/PropXplain)]

## Overview

This repository contains the official implementation for **PropXplain**, a framework for explainable propaganda detection in text. The system performs both classification (propagandistic vs. not-propagandistic) and generates natural language explanations for its predictions. This work is based on the paper "[PropXplain: Can LLMs Enable Explainable Propaganda Detection?](https://aclanthology.org/2025.findings-emnlp.1296/)" published in Findings of EMNLP 2025.

![Experimental Pipeline](assets/Figure1.png)
*Figure 1: Example of a news sentence and its explanation and quality assessment process.*

## Features

- 🎯 **Classification & Explanation**: Detects propaganda and generates detailed explanations.
- 🌐 **Multilingual Support**: Supports Arabic and English.
- 🤖 **LLM-based**: Uses Llama-3.1-8B-Instruct for both tasks.
- 📊 **New Datasets**: Introduces new explanation-enhanced datasets for Arabic and English.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/firojalam/PropXplain.git
cd PropXplain
conda create -n propxplain python=3.10 -y
conda activate propxplain
pip install -r bin/text-classification/requirements.txt

# 2. Configure API keys (for explanation generation)
cp .env.example .env
# Edit .env and add your API keys

# 3. Run inference with a pre-trained model
bash scripts/inference/run_inference_ar.sh

# 4. Evaluate results
bash scripts/evaluation/run_evaluation_ar.sh
```

## Supported Models

| Model | Size | Tasks |
|-------|------|-------|
| **Llama-3.1-Instruct** | 8B | Classification, Explanation |

## Datasets

📊 **Available on HuggingFace:** [QCRI/PropXplain](https://huggingface.co/datasets/QCRI/PropXplain)

### Arabic Propaganda Dataset
- **Size**: ~21K items (18,452 train, 1,318 dev, 1,326 test)
- **Language**: Arabic
- **Labels**: `propagandistic`, `not-propagandistic`
- **Format**: JSONL with explanations
- **Sources**: 300 news agencies + Twitter data
- **Topics**: Politics, human rights, Israeli-Palestinian conflict, and more

### English Propaganda Dataset
- **Size**: ~6K items (4,472 train, 621 dev, 922 test)
- **Language**: English
- **Labels**: `propagandistic`, `not-propagandistic`
- **Format**: JSONL with explanations
- **Sources**: 42 news sources across political spectrum
- **Topics**: Politics, war coverage, trending topics (late 2023-early 2024)

Both datasets include:
- Original text content
- Binary propaganda labels
- Human-validated, LLM-generated explanations
- Quality scores (informativeness, clarity, plausibility, faithfulness)

### Loading the Dataset

```python
from datasets import load_dataset

# Load Arabic dataset
dataset_ar = load_dataset("QCRI/PropXplain", "arabic")

# Load English dataset
dataset_en = load_dataset("QCRI/PropXplain", "english")

# Access splits
train_data = dataset_ar["train"]
dev_data = dataset_ar["validation"]
test_data = dataset_ar["test"]

# Example: iterate through samples
for example in train_data:
    print(f"Text: {example['input']}")
    print(f"Label: {example['label']}")
    print(f"Explanation: {example['explanation']}")
```

## Installation

```bash
# Clone repository
git clone https://github.com/firojalam/PropXplain.git
cd PropXplain

# Create and activate environment
conda create -n propxplain python=3.10 -y
conda activate propxplain

# Install dependencies
pip install -r bin/text-classification/requirements.txt

# For explanation generation (optional)
pip install openai anthropic langchain langchain-openai python-dotenv
```

## Configuration

Before running the scripts, you need to set up your environment variables for API access:

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
# For Azure OpenAI (used in explanation generation)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# For OpenAI (alternative)
OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic Claude (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Note**: The `.env` file is git-ignored and will not be committed to the repository.

## Data Preparation

The raw data needs to be converted into a format suitable for training. The scripts for data processing can be found in `bin/data_processing/`.

```bash
# Format the Arabic data
bash scripts/data_preparation/run_format_data.sh
```

**Available data processing scripts:**
- `convert_to_json.py`: Convert raw data to JSONL format
- `reformat_datasets.py`: Reformat datasets for different tasks
- `merge_predictions.py`: Merge model predictions with original data
- `gpt_output_formatter.py`: Format GPT outputs for evaluation

## Generating Explanations

We used OpenAI's o1 model to generate explanations. The explanation generation scripts are in `bin/augment/`.

**Important**: Set up your API keys in `.env` before running these scripts.

```bash
# Generate explanations for a dataset
python bin/augment/gpt_explainer_v2.py \
    --input_file data/dataset.jsonl \
    --output_file data/dataset_with_explanations.jsonl

# Estimate API costs before running
python bin/augment/estimate_explain_cost.py \
    --input_file data/dataset.jsonl
```

## Training

The model is fine-tuned using LoRA (Low-Rank Adaptation). Training scripts are located in `scripts/training/`.

```bash
# Run training for the Arabic model
bash scripts/training/run_train_ar.sh
```

### Training Parameters

Our configuration uses:
- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Training Method**: LoRA (Parameter-Efficient Fine-Tuning)
- **LoRA Rank**: 8
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.005
- **Learning Rate**: 1e-4
- **Epochs**: 24
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 16 steps
- **Precision**: bfloat16 (bf16)

The training was conducted on 8 NVIDIA H100 GPUs using Distributed Data Parallel (DDP).

## Inference

To run inference with a fine-tuned model, use the scripts in `scripts/inference/`.

```bash
# Run inference for the Arabic model
bash scripts/inference/run_inference_ar.sh
```

## Evaluation

Evaluate the predictions to compute classification and explanation quality metrics.

```bash
# Evaluate the Arabic predictions
bash scripts/evaluation/run_evaluation_ar.sh
```

## Repository Structure

```
PropXplain/
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── README.md                       # This file
├── assets/
│   └── Figure1.png                # Paper figure
├── bin/
│   ├── augment/                   # Explanation generation scripts
│   │   ├── agent.py              # Multi-agent explanation system
│   │   ├── agent_binary.py       # Binary classification agent
│   │   ├── generate_instructions.py  # Instruction generation
│   │   ├── gpt_explainer.py      # GPT-based explainer
│   │   └── gpt_explainer_v2.py   # Enhanced explainer
│   ├── data_processing/           # Data formatting scripts
│   │   ├── convert_to_json.py
│   │   ├── reformat_datasets.py
│   │   └── merge_predictions.py
│   ├── text-classification/       # Training & inference scripts
│   │   ├── llm_training.py       # LoRA fine-tuning
│   │   ├── llm_inference.py      # Model inference
│   │   ├── run_inference.py      # Batch inference
│   │   ├── exp_utils.py          # Evaluation utilities
│   │   └── requirements.txt      # Python dependencies
│   └── train/                     # Additional training utilities
└── scripts/                       # Example shell scripts
    ├── data_preparation/
    │   └── run_format_data.sh
    ├── training/
    │   └── run_train_ar.sh
    ├── inference/
    │   └── run_inference_ar.sh
    └── evaluation/
        └── run_evaluation_ar.sh
```

**Note**: The `data/` directory is not included in the repository. Please refer to the paper or contact the authors for dataset access.

## Key Results

Our experiments with Llama-3.1-8B-Instruct show:

### Arabic Dataset
- **Micro F1**: 0.775 (comparable to AraBERT baseline: 0.762)
- **Macro F1**: 0.760
- **BERTScore F1**: 0.706 for explanations

### English Dataset
- **Micro F1**: 0.781 (outperforms BERT-base: 0.772)
- **Macro F1**: 0.675
- **BERTScore F1**: 0.751 for explanations

The model achieves competitive classification performance while also generating high-quality explanations validated by human evaluators.

## Explanation Quality

Human evaluation of generated explanations (5-point Likert scale):

| Metric | Arabic | English |
|--------|--------|---------|
| **Faithfulness** | 4.35 | 4.72 |
| **Clarity** | 4.49 | 4.76 |
| **Plausibility** | 4.42 | 4.71 |
| **Informativeness** | 4.26 | 4.71 |

High inter-annotator agreement: r*wg(j) > 0.89 (Arabic), > 0.94 (English)

## Contact

For questions, dataset access, or collaboration inquiries:
- **Firoj Alam**: fialam@hbku.edu.qa
- **Giovanni Da San Martino**: giovanni.dasanmartino@unipd.it

For issues or bug reports, please use the [GitHub Issues](https://github.com/firojalam/PropXplain/issues) page.

## Citation

If you use our resources, please cite:
```bibtex
@inproceedings{hasanain-etal-2025-propxplain,
    title = "{P}rop{X}plain: Can {LLM}s Enable Explainable Propaganda Detection?",
    author = "Hasanain, Maram  and
      Hasan, Md Arid  and
      Kmainasi, Mohamed Bayan  and
      Sartori, Elisa  and
      Shahroor, Ali Ezzat  and
      Da San Martino, Giovanni  and
      Alam, Firoj",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1296/",
    pages = "23855--23863",
    ISBN = "979-8-89176-335-7",
    abstract = "There has been significant research on propagandistic content detection across different modalities and languages. However, most studies have primarily focused on detection, with little attention given to explanations justifying the predicted label. This is largely due to the lack of resources that provide explanations alongside annotated labels. To address this issue, we propose a multilingual (i.e., Arabic and English) explanation-enhanced dataset, the first of its kind. Additionally, we introduce an explanation-enhanced LLM for both label detection and rationale-based explanation generation. Our findings indicate that the model performs comparably while also generating explanations. We will make the dataset and experimental resources publicly available for the research community (https://github.com/firojalam/PropXplain)."
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

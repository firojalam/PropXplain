import optparse
import os.path
from datasets import Dataset
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix

import logging

from transformers import (
    DataCollatorForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

from XLMRobertaForSpanCategorization import XLMRobertaForSpanCategorization
from BertForSpanCategorization import BertForSpanCategorization

random_seed = 12345


def load_ds(train_fname, dev_fname):
    train_ds = Dataset.from_json(train_fname)
    logger.info("Train sample: " + str(train_ds[0]))

    train_ds = train_ds.shuffle(seed=random_seed)  # To shuffle rows in case when we have multiple train sets
    dev_ds = Dataset.from_json(dev_fname)
    logger.info("Dev sample: " + str(dev_ds[0]))

    logger.info("Loaded %d training examples..." % len(train_ds))

    return train_ds, dev_ds


def load_tech_list(fname):
    label2id = {}
    label2id["O"] = 0

    i = 1
    with open(fname, 'r') as inf:
        for line in inf:
            if "no_technique" in line: continue
            if "no technique" in line: continue
            if "no-technique" in line: continue

            tech = "B-" + line.strip()
            itech = "I-" + line.strip()

            label2id[tech] = i
            label2id[itech] = i + 1

            i += 2

    id2label = {v: k for k, v in label2id.items()}

    logger.info(label2id)
    logger.info(id2label)

    return id2label, label2id


def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int):
    """
    Check if the token is inside a span.
    Args:
      - token_start, token_end: Start and end offset of the token
      - span_start, span_end: Start and end of the span
    Returns:
      - "B" if beginning
      - "I" if inner
      - "O" if outer
      - "N" if not valid token (like <SEP>, <CLS>, <UNK>)
    """
    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start > span_start:
        return "I"
    else:
        return "B"


def tokenize_and_adjust_labels(sample, label2id, max_seq_len, tokenizer):
    """
    Args:
        - sample (dict): {"id": "...", "paragraph": "...", "labels": [{"start": ..., "end": ..., "technique": ...}, ...]
    Returns:
        - The tokenized version of `sample` and the labels of each token.
    """
    # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
    # Use max_length and truncation to ajust the text length
    tokenized = tokenizer(sample["paragraph"] if 'paragraph' in sample else sample["text"],
                          return_offsets_mapping=True,
                          padding="max_length",
                          max_length=max_seq_len,
                          truncation=True)

    # We are doing a multilabel classification task at each token, we create a list of size len(label2id)
    # for the labels
    labels = [[0 for _ in label2id.keys()] for _ in range(max_seq_len)]

    # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
    # or inside the spans
    for (token_start, token_end), token_labels in zip(tokenized["offset_mapping"], labels):
        for span in sample["labels"]:
            # Assign label "Other" to techniques we didn't load with data to decrease label space
            if f"B-{span['technique']}" not in label2id:
                span['technique'] = "Other"

            role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
            if role == "B":
                token_labels[label2id[f"B-{span['technique']}"]] = 1
            elif role == "I":
                token_labels[label2id[f"I-{span['technique']}"]] = 1

    return {**tokenized, "labels": labels}


def tokenize_ds(train_ds, dev_ds, label2id, max_seq_len, tokenizer):
    tokenized_train_ds = [tokenize_and_adjust_labels(sample, label2id, max_seq_len, tokenizer) for sample in train_ds]
    tokenized_dev_ds = [tokenize_and_adjust_labels(sample, label2id, max_seq_len, tokenizer) for sample in dev_ds]

    # to print example of formatted sample
    sample = tokenized_train_ds[0]
    logger.info("--------Token---------|--------Labels----------")
    for token_id, token_labels in zip(sample["input_ids"], sample["labels"]):
        # Decode the token_id into text
        token_text = tokenizer.decode(token_id)

        # Retrieve all the indices corresponding to the "1" at each token, decode them to label name
        labels = [id2label[label_index] for label_index, value in enumerate(token_labels) if value == 1]

        # Decode those indices into label name
        logger.info(f" {token_text:20} | {labels}")

        # Finish when we meet the end of sentence.
        if token_text == "</s>":
            break

    return tokenized_train_ds, tokenized_dev_ds


def divide(a: int, b: int):
    return a / b if b > 0 else 0


def compute_metrics(p):
    """
    Customize the `compute_metrics` of `transformers`
    Args:
        - p (tuple):      2 numpy arrays: predictions and true_labels
    Returns:
        - metrics (dict): f1 score on
    """
    # (1)
    predictions, true_labels = p

    # (2)
    predicted_labels = np.where(predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape))
    metrics = {}

    # (3)
    cm = multilabel_confusion_matrix(true_labels.reshape(-1, n_labels), predicted_labels.reshape(-1, n_labels))

    # (4)
    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue  # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{id2label[label_idx]}"] = f1

    # (5)
    macro_f1 = sum(list(metrics.values())) / (n_labels - 1)
    metrics["macro_f1"] = macro_f1

    return metrics


def train(base_model_name, out_dir, tokenizer, tokenized_train_ds, tokenized_dev_ds, training_args,
          data_collator):
    if "xlm-roberta" in base_model_name:
        model = XLMRobertaForSpanCategorization.from_pretrained(base_model_name, id2label=id2label, label2id=label2id)
    elif "bert-base-arabic" in base_model_name or "arabert" in base_model_name:
        model = BertForSpanCategorization.from_pretrained(base_model_name, id2label=id2label, label2id=label2id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    trainer.tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-t', '--train_file', action="store", dest="train_file", default=None, type="string",
                      help='Train file path')
    parser.add_option('-d', '--dev_file', action="store", dest="dev_file", default=None, type='string',
                      help="Dev file path.")
    parser.add_option('-o', '--out_dir', action='store', dest='out_dir', default=None, type="string",
                      help='Directory to keep outputs.')
    parser.add_option('-q', '--tech_file', action='store', dest='tech_file', default=None, type="string",
                      help='Techniques labels to keep in train and test.')
    parser.add_option('-m', '--model', action='store', dest='model', default=None, type="string",
                      help='Base transformer model to train.')
    parser.add_option('-e', '--epochs', action='store', dest='epochs', default=None, type="int",
                      help='Number of train epochs.')
    parser.add_option('-l', '--max_seq_len', action='store', dest='max_seq_len', default=None, type="int",
                      help='Number of train epochs.')

    options, args = parser.parse_args()

    train_fname = options.train_file
    dev_fname = options.dev_file
    out_dir = options.out_dir
    tech_file = options.tech_file
    epochs = options.epochs
    max_seq_len = options.max_seq_len
    base_model = options.model

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO,
                        datefmt='%dd/%mm %H:%M:%S')

    if os.path.isdir(out_dir):
        logger.info("Model already trained, will skip\t" + str(out_dir))
        exit()

    train_ds, dev_ds = load_ds(train_fname, dev_fname)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    id2label, label2id = load_tech_list(tech_file) # This should be the techniques list we will keep (not just the 23 ones)
    n_labels = len(id2label)

    tokenized_train_ds, tokenized_dev_ds = tokenize_ds(train_ds, dev_ds, label2id, max_seq_len, tokenizer)

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        eval_steps=100,
        learning_rate=1.0e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.001,
        logging_steps=500,
        save_strategy='epoch',
        load_best_model_at_end=True,  # this will let the model save the best checkpoint
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        log_level='critical',

        seed=random_seed
    )

    train(base_model, out_dir, tokenizer, tokenized_train_ds, tokenized_dev_ds, training_args, data_collator)

    logger.info("Done training! Model saved to: " + str(out_dir))

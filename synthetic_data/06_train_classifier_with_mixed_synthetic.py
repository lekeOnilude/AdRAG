"""
Train a new ad-classifier with the mix of touche, naivesynthetic, and structuredsynthetic
"""

import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
)
import torch
from torch.utils.data import Dataset
import random
import math

# Optional Weights & Biases integration
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False
    print(
        "Weights & Biases not installed. To use wandb reporting, install with: pip install wandb"
    )


random.seed(42)

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SYNTH_DATA_DIR_PATH = os.path.join(CUR_DIR_PATH, "data")
# Synthetic data
SYNTH_NAIVE_FP = os.path.join(SYNTH_DATA_DIR_PATH, "single-prompt-multi-model.jsonl")
SYNTH_STRUCTURED_FP = os.path.join(SYNTH_DATA_DIR_PATH, "synthetic-all.jsonl")
SYNTH_STRUCTURED_LABELS_FP = os.path.join(
    SYNTH_DATA_DIR_PATH, "synthetic-all-labels.jsonl"
)
# Touche data
TOUCHE_DATA_DIR_PATH = os.path.join(os.path.dirname(CUR_DIR_PATH), "data", "subtask-2")
# Output directory for trained model
OUTPUT_DIR = os.path.join(CUR_DIR_PATH, "models", "ad-classifier-v0.3")
# Directory for logs and visualization
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Parse the labels and responses from the original jsonl files
def parse_responses_to_dict(jsonl_path):
    response_to_dict = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing responses"):
            data = json.loads(line)
            response_to_dict[data["id"]] = data["response"]
    return response_to_dict


def parse_labels_to_dict(jsonl_path):
    query_id_to_label = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing labels"):
            data = json.loads(line)
            query_id_to_label[data["id"]] = int(data["label"])
    return query_id_to_label


def parse_synthetic_naive_data(jsonl_path):
    response_to_dict = {}
    query_id_to_label = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing synthetic responses"):
            data = json.loads(line)
            qid_ad = f"SYNTH-{data['query_id']}-A"
            qid_no_ad = f"SYNTH-{data['query_id']}-N"
            response_to_dict[qid_ad] = data["with_ad"]
            response_to_dict[qid_no_ad] = data["without_ad"]
            query_id_to_label[qid_ad] = 1
            query_id_to_label[qid_no_ad] = 0

    return response_to_dict, query_id_to_label


## 1) initializing train and validation set with touche data
train_responses: dict = parse_responses_to_dict(
    os.path.join(TOUCHE_DATA_DIR_PATH, "responses-train.jsonl")
)
train_label: dict = parse_labels_to_dict(
    os.path.join(TOUCHE_DATA_DIR_PATH, "responses-train-labels.jsonl")
)

valid_responses: dict = parse_responses_to_dict(
    os.path.join(TOUCHE_DATA_DIR_PATH, "responses-validation.jsonl")
)
valid_label: dict = parse_labels_to_dict(
    os.path.join(TOUCHE_DATA_DIR_PATH, "responses-validation-labels.jsonl")
)

## 2) Expanding train and validation set with NaiveSynthetic dataset
synth_naive_responses, synth_naive_labels = parse_synthetic_naive_data(SYNTH_NAIVE_FP)

synth_naive_keys = list(synth_naive_responses.keys())
num_valid_naive = math.ceil(len(synth_naive_keys) * 0.2)
random.shuffle(synth_naive_keys)
synth_naive_valid_keys = synth_naive_keys[:num_valid_naive]
synth_naive_train_keys = synth_naive_keys[num_valid_naive:]

for key in synth_naive_train_keys:
    train_responses[key] = synth_naive_responses[key]
    train_label[key] = synth_naive_labels[key]

for key in synth_naive_valid_keys:
    valid_responses[key] = synth_naive_responses[key]
    valid_label[key] = synth_naive_labels[key]

print(
    f"NaiveSynthetic examples: {len(synth_naive_keys)} total, {len(synth_naive_train_keys)} added to train, {len(synth_naive_valid_keys)} added to validation."
)
print(f"Total training examples so far: {len(train_responses)}")
print(f"Total validation examples so far: {len(valid_responses)}\n")


## 3) Expanding train and validation set with StructuredSynthetic dataset
synth_struct_responses = parse_responses_to_dict(SYNTH_STRUCTURED_FP)
synth_struct_labels = parse_labels_to_dict(SYNTH_STRUCTURED_LABELS_FP)

synth_struct_keys = list(synth_struct_responses.keys())
num_valid_struct = math.ceil(len(synth_struct_keys) * 0.1)
random.shuffle(synth_struct_keys)
synth_struct_valid_keys = synth_struct_keys[:num_valid_struct]
synth_struct_train_keys = synth_struct_keys[num_valid_struct:]

for key in synth_struct_train_keys:
    train_responses[key] = synth_struct_responses[key]
    train_label[key] = synth_struct_labels[key]

for key in synth_struct_valid_keys:
    valid_responses[key] = synth_struct_responses[key]
    valid_label[key] = synth_struct_labels[key]

print(
    f"StructuredSynthetic examples: {len(synth_struct_keys)} total, {len(synth_struct_train_keys)} added to train, {len(synth_struct_valid_keys)} added to validation."
)
print(f"Total training examples so far: {len(train_responses)}")
print(f"Total validation examples so far: {len(valid_responses)}")


## 4) Train
class ClassificationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [{"input_ids": f["input_ids"]} for f in features]

        example = self.tokenizer.pad(
            input_ids,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        example["labels"] = torch.tensor(
            [int(feature["label"]) for feature in features]
        )

        return example


class AdvertisementDataset(Dataset):
    def __init__(self, responses_dict, labels_dict, tokenizer):
        self.responses_dict = responses_dict
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.ids = list(responses_dict.keys())
        self.show_random_example()

    def show_random_example(self):
        qid = np.random.randint(0, len(self))
        print(f"Random example {qid}:")
        print("LLM response to query:", self.responses_dict[self.ids[qid]])
        print("---------------------")
        print("Contains ad?", self.labels_dict[self.ids[qid]])
        print("#####################")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        query_id = self.ids[item]
        sentence = self.responses_dict[query_id]
        label = self.labels_dict[query_id]

        tokenized_sentence = self.tokenizer(
            sentence,
            padding=False,
            truncation=True,
            max_length=512,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        tokenized_sentence["label"] = label

        return tokenized_sentence


# Train the model

model_to_train = "microsoft/deberta-v3-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_to_train)

train_dataset = AdvertisementDataset(
    responses_dict=train_responses, labels_dict=train_label, tokenizer=tokenizer
)
valid_dataset = AdvertisementDataset(
    responses_dict=valid_responses, labels_dict=valid_label, tokenizer=tokenizer
)


collator = ClassificationCollator(tokenizer)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()

    # Calculate more detailed metrics
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=500,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    # fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
    # gradient_accumulation_steps=2,  # Accumulate gradients to handle larger batch sizes
    # warmup_steps=500,  # Warm up learning rate for stability
    report_to=["wandb"] if wandb_available else None,
)

# Initialize wandb if available
if wandb_available:
    try:
        wandb.init(
            project="advertisement-classifier",
            name=f"ad-classifier-v0.3-{model_to_train.split('/')[-1]}",
            config={
                "model": model_to_train,
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "weight_decay": training_args.weight_decay,
            },
        )
        print("Weights & Biases initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Weights & Biases: {e}")
        wandb_available = False

model = AutoModelForSequenceClassification.from_pretrained(model_to_train, num_labels=2)
model.to(device)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# Display final evaluation metrics
final_metrics = trainer.evaluate()
print(f"Final evaluation metrics: {final_metrics}")

# Finish wandb run if active
if wandb_available and wandb.run is not None:
    # Log final metrics to wandb
    wandb.log(final_metrics)
    print("Final metrics logged to Weights & Biases.")

    # Log model artifacts if desired
    # wandb.save(os.path.join(OUTPUT_DIR, "*.bin"))  # Uncomment to save model weights

    # Close wandb run
    wandb.finish()
    print("Weights & Biases tracking completed.")

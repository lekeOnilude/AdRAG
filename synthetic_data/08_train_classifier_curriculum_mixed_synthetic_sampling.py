"""
For curriculumn learning, we rank the data by the classification confidence (by logit value)
We also upsample touche and structured sythetic data and downsample naive synthetic data
"""
import os
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
)
from torch.utils.data import Dataset
import math
import random

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
SYNTH_STRUCTURED_FP = os.path.join(
    SYNTH_DATA_DIR_PATH, "synthetic-structured-all.jsonl"
)
SYNTH_STRUCTURED_LABELS_FP = os.path.join(
    SYNTH_DATA_DIR_PATH, "synthetic-structured-all-labels.jsonl"
)
# Touche data
TOUCHE_DATA_DIR_PATH = os.path.join(os.path.dirname(CUR_DIR_PATH), "data", "subtask-2")
# Output directory for trained model
OUTPUT_DIR = os.path.join(CUR_DIR_PATH, "models", "ad-classifier-v0.5")
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "curriculum_train")
OUTPUT_DIR_CURRICULUM_TRAIN_FP = os.path.join(OUTPUT_TRAIN_DIR, "train.jsonl")
OUTPUT_DIR_CURRICULUM_TRAIN_LABELS_FP = os.path.join(
    OUTPUT_TRAIN_DIR, "train-labels.jsonl"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def setup_model(hf_model_path: str):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
    model.eval()
    model.to(device)
    return tokenizer, model


def get_logit_score_of_correct_label(
    tokenizer, model, passages: list[str], labels: list[int], batch_size=32
):
    """
    Get the logit score of the correct label for each passage.

    Args:
        tokenizer: The model tokenizer
        model: The classification model
        passages: List of text passages to classify
        labels: List of correct labels corresponding to passages
        batch_size: Number of passages to process at once

    Returns:
        List of logit scores for the correct label of each passage
    """
    all_logit_scores = []

    # Process in batches
    for i in tqdm(range(0, len(passages), batch_size), desc="Classification batches"):
        batch_passages = passages[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        # Clear CUDA cache before processing each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = tokenizer(
            batch_passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # For each sample in the batch, get the logit score for the correct label
            batch_scores = []
            for j, label in enumerate(batch_labels):
                # Extract the logit score for the correct class (0 or 1)
                correct_class_logit = logits[j, label].item()
                batch_scores.append(correct_class_logit)

            all_logit_scores.extend(batch_scores)

        # Remove references to tensors
        del inputs, outputs, logits

    return all_logit_scores


def rank_and_save_training_data(
    train_responses, train_label, tokenizer, model, output_path, output_labels_path
):
    """
    Rank training data by difficulty based on logit scores and save to file.

    Lower logit scores for correct class indicate more difficult examples.
    """
    print("Preparing data for ranking...")

    # Extract IDs, responses, and labels
    ids = list(train_responses.keys())
    responses = [train_responses[id] for id in ids]
    labels = [train_label[id] for id in ids]

    print(f"Computing logit scores for {len(responses)} training examples...")

    # Get logit scores for correct labels
    logit_scores = get_logit_score_of_correct_label(
        tokenizer, model, responses, labels, batch_size=32
    )

    # Create a list of (id, response, label, logit_score) tuples
    ranked_data = list(zip(ids, responses, labels, logit_scores))

    # Sort by logit score (descending) - lower scores mean more difficult examples
    ranked_data.sort(key=lambda x: x[3], reverse=True)

    print("Saving curriculum learning data...")

    # Save responses and labels simultaneously
    with open(output_path, "w", encoding="utf-8") as f_responses, open(
        output_labels_path, "w", encoding="utf-8"
    ) as f_labels:
        for id, response, label, logit_score in ranked_data:
            # Write to responses file
            response_obj = {"id": id, "response": response}
            f_responses.write(json.dumps(response_obj) + "\n")

            # Write to labels file
            label_obj = {"id": id, "label": label, "logit": logit_score}
            f_labels.write(json.dumps(label_obj) + "\n")

    print(f"Ranked data saved to {output_path} and {output_labels_path}")

    # Create sorted dictionaries to return
    sorted_responses_dict = {}
    sorted_labels_dict = {}

    for id, response, label, _ in ranked_data:
        sorted_responses_dict[id] = response
        sorted_labels_dict[id] = label

    # Return statistics about the ranking
    difficulty_stats = {
        "easiest_logit": ranked_data[0][3],
        "hardest_logit": ranked_data[-1][3],
        "median_logit": ranked_data[len(ranked_data) // 2][3],
        "mean_logit": sum(item[3] for item in ranked_data) / len(ranked_data),
    }

    return difficulty_stats, sorted_responses_dict, sorted_labels_dict


def sample_data(responses_dict, labels_dict, ratio=1.0, note=None, downsample=False):
    """
    Sample data from responses and labels dictionaries.

    Args:
        responses_dict: Dictionary mapping IDs to response texts
        labels_dict: Dictionary mapping IDs to labels
        ratio: Sampling ratio (>1 for upsampling, <1 for downsampling)
        note: String to append to sampled IDs to distinguish them
        downsample: If True, perform downsampling; if False, perform upsampling

    Returns:
        Tuple of (sampled_responses_dict, sampled_labels_dict)
    """
    if ratio == 1.0:
        return responses_dict.copy(), labels_dict.copy()

    original_ids = list(responses_dict.keys())
    sampled_responses = responses_dict.copy()
    sampled_labels = labels_dict.copy()

    if downsample:
        # Downsampling: randomly select a subset of the original data
        num_to_keep = int(len(original_ids) * ratio)
        ids_to_keep = random.sample(original_ids, num_to_keep)

        # Create new dictionaries with only the selected IDs
        downsampled_responses = {id: responses_dict[id] for id in ids_to_keep}
        downsampled_labels = {id: labels_dict[id] for id in ids_to_keep}

        return downsampled_responses, downsampled_labels
    else:
        # Upsampling: duplicate existing examples
        num_to_add = int(len(original_ids) * (ratio - 1))

        # For upsampling, we'll randomly select IDs to duplicate
        if num_to_add > 0:
            # We may need to select IDs multiple times if ratio is high
            ids_to_duplicate = random.choices(original_ids, k=num_to_add)

            # Add duplicated examples with modified IDs
            for i, original_id in enumerate(ids_to_duplicate):
                new_id = f"{original_id}{note}-{i}"
                sampled_responses[new_id] = responses_dict[original_id]
                sampled_labels[new_id] = labels_dict[original_id]

        return sampled_responses, sampled_labels


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

# Save original touche counts for reporting
touche_train_count = len(train_responses)
print(f"Original Touche training examples: {touche_train_count}")


# Upsample touche training data with ratio=3
train_responses, train_label = sample_data(
    train_responses, train_label, ratio=3, note="-upsampled"
)
print(f"After upsampling, Touche training examples: {len(train_responses)}")

## 2) Expanding train and validation set with NaiveSynthetic dataset
synth_naive_responses, synth_naive_labels = parse_synthetic_naive_data(SYNTH_NAIVE_FP)

synth_naive_keys = list(synth_naive_responses.keys())
num_valid_naive = math.ceil(len(synth_naive_keys) * 0.2)
random.shuffle(synth_naive_keys)
synth_naive_valid_keys = synth_naive_keys[:num_valid_naive]
synth_naive_train_keys = synth_naive_keys[num_valid_naive:]

print(f"Original naive synthetic training examples: {len(synth_naive_train_keys)}")

# Downsample synthetic naive train dataset with sampling ratio of 0.07
naive_train_responses = {
    key: synth_naive_responses[key] for key in synth_naive_train_keys
}
naive_train_labels = {key: synth_naive_labels[key] for key in synth_naive_train_keys}

downsampled_naive_responses, downsampled_naive_labels = sample_data(
    naive_train_responses, naive_train_labels, ratio=0.7, downsample=True
)
print(
    f"After downsampling, naive synthetic training examples: {len(downsampled_naive_responses)}"
)

# Add downsampled naive synthetic data to training sets
for key in downsampled_naive_responses:
    train_responses[key] = downsampled_naive_responses[key]
    train_label[key] = downsampled_naive_labels[key]

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

print(
    f"Original structured synthetic training examples: {len(synth_struct_train_keys)}"
)

# Extract structured synthetic training data
struct_train_responses = {
    key: synth_struct_responses[key] for key in synth_struct_train_keys
}
struct_train_labels = {key: synth_struct_labels[key] for key in synth_struct_train_keys}

# Upsample synthetic structured train dataset with ratio of 3
upsampled_struct_responses, upsampled_struct_labels = sample_data(
    struct_train_responses, struct_train_labels, ratio=3, note="-upsampled"
)

print(
    f"After upsampling, structured synthetic training examples: {len(upsampled_struct_responses)}"
)

# Add upsampled structured synthetic data to training sets
for key in upsampled_struct_responses:
    train_responses[key] = upsampled_struct_responses[key]
    train_label[key] = upsampled_struct_labels[key]

for key in synth_struct_valid_keys:
    valid_responses[key] = synth_struct_responses[key]
    valid_label[key] = synth_struct_labels[key]

print(
    f"StructuredSynthetic examples: {len(synth_struct_keys)} total, {len(synth_struct_train_keys)} added to train, {len(synth_struct_valid_keys)} added to validation."
)
print(f"Total training examples so far: {len(train_responses)}")
print(f"Total validation examples so far: {len(valid_responses)}")


##################
# Rank data entries by their difficulty
##################
tokenizer, model = setup_model("jmvcoelho/ad-classifier-v0.1")

# Sort training example by difficulty and save the training data to output file
#   difficulty: you check the logit score of the correct label. smaller the logit score indicates harder example.
stats, train_responses, train_label = rank_and_save_training_data(
    train_responses,
    train_label,
    tokenizer,
    model,
    OUTPUT_DIR_CURRICULUM_TRAIN_FP,
    OUTPUT_DIR_CURRICULUM_TRAIN_LABELS_FP,
)

print("\nLogit Scores Statistics:")
print(f"Easiest example (highest logit score): {stats['easiest_logit']:.4f}")
print(f"Hardest example (lowest logit score): {stats['hardest_logit']:.4f}")
print(f"Median difficulty: {stats['median_logit']:.4f}")
print(f"Mean difficulty: {stats['mean_logit']:.4f}")

print("\nData prepared for curriculum learning training!")


##################
# Train a classifier
##################
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
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=1000,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=5000,
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
            name=f"ad-classifier-v0.4-{model_to_train.split('/')[-1]}",
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
    wandb.save(os.path.join(OUTPUT_DIR, "*.bin"))  # Uncomment to save model weights

    # Close wandb run
    wandb.finish()
    print("Weights & Biases tracking completed.")

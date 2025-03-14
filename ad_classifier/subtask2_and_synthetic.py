from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
import json
import random
import math


# Parse the labels and responses from the original jsonl files
def parse_labels_to_dict(jsonl_path):
    query_id_to_label = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing labels"):
            data = json.loads(line)
            query_id_to_label[data["id"]] = int(data["label"])
    return query_id_to_label


def parse_responses_to_dict(jsonl_path):
    response_to_dict = {}
    with open(jsonl_path, "r") as jsonl_file:
        for line in tqdm(jsonl_file, desc="Processing responses"):
            data = json.loads(line)
            response_to_dict[data["id"]] = data["response"]
    return response_to_dict


def parse_synthetic_data(jsonl_path):
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



train_labels_jsonl_path = "/home/jmcoelho/11797_Project/data/subtask-2/responses-train-labels.jsonl"
train_responses_jsonl_path = "/home/jmcoelho/11797_Project/data/subtask-2/responses-train.jsonl"
train_label = parse_labels_to_dict(train_labels_jsonl_path)
train_responses = parse_responses_to_dict(train_responses_jsonl_path)

valid_labels_jsonl_path = "/home/jmcoelho/11797_Project/data/subtask-2/responses-validation-labels.jsonl"
valid_responses_jsonl_path = "/home/jmcoelho/11797_Project/data/subtask-2/responses-validation.jsonl"
valid_label = parse_labels_to_dict(valid_labels_jsonl_path)
valid_responses = parse_responses_to_dict(valid_responses_jsonl_path)

# combine original data with new syntetic data.
synthetic_responses, synthetic_labels = parse_synthetic_data("/home/jmcoelho/11797_Project/rewriter/output/marcov2.train/Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2/Qwen2.5-7B-Instruct-10-passage-RAG/prompt1-12/all.jsonl")

synthetic_keys = list(synthetic_responses.keys())
num_total = len(synthetic_keys)
num_valid = math.ceil(num_total * 0.1)
random.seed(42)
random.shuffle(synthetic_keys)
synthetic_valid_keys = synthetic_keys[:num_valid]
synthetic_train_keys = synthetic_keys[num_valid:]

for key in synthetic_train_keys:
    train_responses[key] = synthetic_responses[key]
    train_label[key] = synthetic_labels[key]

for key in synthetic_valid_keys:
    valid_responses[key] = synthetic_responses[key]
    valid_label[key] = synthetic_labels[key]

print(f"Synthetic examples: {num_total} total, {len(synthetic_train_keys)} added to train, {len(synthetic_valid_keys)} added to validation.")
print(f"Total training examples: {len(train_responses)}")
print(f"Total validation examples: {len(valid_responses)}")

# Huggingface dataset interface


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

out_dir = "./models/subtask2_with_synth_v2"

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
    return {"accuracy": accuracy}


training_args = TrainingArguments(
    output_dir=out_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    logging_steps=1,
    report_to="wandb",
)

model = AutoModelForSequenceClassification.from_pretrained(model_to_train, num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(out_dir)

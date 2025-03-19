import os
import json
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.join(CUR_DIR_PATH, "data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_model(hf_model_path: str):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
    model.eval()
    model.to(device)
    return tokenizer, model


def classify(tokenizer, model, passages: list[str], batch_size=32):
    """
    Classify passages in batches to avoid CUDA out of memory errors.

    Args:
        tokenizer: The model tokenizer
        model: The classification model
        passages: List of text passages to classify
        batch_size: Number of passages to process at once

    Returns:
        List of predictions
    """
    all_predictions = []

    # Process in batches
    for i in tqdm(range(0, len(passages), batch_size), desc="Classification batches"):
        batch_passages = passages[i : i + batch_size]

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
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().tolist())

        # Remove references to tensors
        del inputs, outputs, logits, predictions

    return all_predictions


def evaluate(preds, labels):
    # Compute F1 score, accuracy, precision, and recall
    f1 = f1_score(labels, preds, average="binary")
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    return {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_path", type=str, default="jmvcoelho/ad-classifier-v0.1"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for classification"
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(CUR_DIR_PATH, "results"), exist_ok=True)
    RESULTS_SAVE_FP = os.path.join(
        CUR_DIR_PATH,
        "results",
        f"{(args.hf_model_path).split('/')[1]}_synthetic-all_results.json",
    )

    tokenizer, model = setup_model(args.hf_model_path)

    # Load dataset files
    responses_data = load_jsonl(os.path.join(DATA_DIR_PATH, "synthetic-all.jsonl"))
    labels_data = load_jsonl(os.path.join(DATA_DIR_PATH, "synthetic-all-labels.jsonl"))

    # Create a mapping from ID to label
    id_to_label = {item["id"]: item["label"] for item in labels_data}

    # Prepare aligned lists of texts and labels
    texts = []
    labels = []

    print(f"Processing {len(responses_data)} entries...")
    for item in tqdm(responses_data):
        item_id = item["id"]
        if item_id in id_to_label:
            texts.append(item["response"])
            labels.append(id_to_label[item_id])

    print(f"Successfully matched {len(texts)} entries with labels")

    # make inferences with the classifier
    print("Running classification...")
    preds = classify(tokenizer, model, texts)

    # get evaluation metrics
    metrics = evaluate(preds, labels)
    print(f"Evaluation metrics: {metrics}")

    # Save metrics to results file
    with open(RESULTS_SAVE_FP, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.hf_model_path,
                "data": "synthetic-all",
                "metrics": metrics,
                "dataset_size": len(texts),
            },
            f,
            indent=2,
        )

    print(f"Results saved to {RESULTS_SAVE_FP}")

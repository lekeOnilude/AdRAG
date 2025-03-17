import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# Model paths
model_dir = "jmvcoelho/ad-classifier-v0.0"  # Path to your trained classification model
directory = "rewriter/rewritten_response_output/" # Path to the rewritten responses

# Load the classification model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda") # Load on GPU if available

# Load rewritten responses
def read_file(file_path):
    rewritten_responses = []
    with open(file_path, "r") as f:
        for line in f:
            # rewritten_responses.append(json.loads(line))
            data = json.loads(line)
            responses = {
                "id": data["id"],
                "generated_text": data["generated_text"]
            }
            rewritten_responses.append(responses)
    return rewritten_responses




# Function to predict the label for a given text
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda") # Move to GPU
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label

def remove_unicode_encode(text):
  return text.encode("ascii", "ignore").decode()

# Predict labels for each rewritten response

for filename in tqdm(os.listdir(directory)):
    if (filename.endswith(".jsonl") or filename.endswith(".json")) and filename.startswith("baseline"):
        filepath = os.path.join(directory, filename)
        rewritten_responses = read_file(filepath)
        predictions = []
        for item in tqdm(rewritten_responses, desc="Predicting labels"):
            text = item["generated_text"]
            predicted_label = predict_label(text)
            predictions.append({"id": item["id"], "predicted_label": predicted_label})

        classifier_type = None
        if model_dir == "jmvcoelho/ad-classifier-v0.1":
            classifier_type = "classifier_v1"
        elif model_dir == "jmvcoelho/ad-classifier-v0.2":
            classifier_type = "classifier_v2"
        elif model_dir == "jmvcoelho/ad-classifier-v0.0":
            classifier_type = "classifier_v0"
        else:
            classifier_type = "unknown"

        output_dir = f"./predictions/dummydata/{classifier_type}" if "dummydata" in directory else f"./predictions/touche_data/{classifier_type}"
        os.makedirs(output_dir, exist_ok=True)

        # Save the predictions
        output_path = '_'.join(filename.split('.')[:-1])
        with open(f"{output_dir}/predicted_label_{output_path}.jsonl", "w") as outfile:
            for prediction in predictions:
                json.dump(prediction, outfile)
                outfile.write("\n")

        print(f"Predictions saved to {output_path}.jsonl")



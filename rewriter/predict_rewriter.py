import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Model paths
model_dir = "./models/subtask2_test"  # Path to your trained classification model
rewritten_responses_file = "rewritten_responses_gemma-7b-it_fewshot_2.json" # Path to the rewritten responses

# Load the classification model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda") # Load on GPU if available

# Load rewritten responses
rewritten_responses = []
with open(rewritten_responses_file, "r") as f:
    for line in f:
        # rewritten_responses.append(json.loads(line))
        data = json.loads(line)
        responses = {
            "id": data["id"],
            "generated_text": data["generated_text"]
        }
        rewritten_responses.append(responses)


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
predictions = []
for item in tqdm(rewritten_responses, desc="Predicting labels"):
    text = item["generated_text"]
    # text = remove_unicode_encode(text)
    predicted_label = predict_label(text)
    predictions.append({"id": item["id"], "predicted_label": predicted_label})


# Save the predictions
with open(f"./predictions/predicted_label{'_'.join(rewritten_responses_file.split('_')[2:]).split('.')[0]}.jsonl", "w") as outfile:
    for prediction in predictions:
        json.dump(prediction, outfile)
        outfile.write("\n")

print("Predictions saved to predicted_labels_rewritten.jsonl")



import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import argparse


def load_classifier(model_dir):
    # Load the classification model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda")
    return model, tokenizer


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


def predict_label(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda") # Move to GPU
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict labels for generated responses.')
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default="jmvcoelho/ad-classifier-v0.0",
        help='classifier model directory on huggingface')
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default="rewriter/rewritten_response_output",
        help='directory containing rewritten responses')
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default="./predictions/touche_data",
        help='directory to save predictions')
    
    args = parser.parse_args()

    model_dir = args.model_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    

    model, tokenizer = load_classifier(model_dir)

    classifier_type = None
    classifier_type = f"classifier_v{model_dir.split('.')[-1]}"

    for filename in tqdm(os.listdir(input_dir)):
        if (filename.endswith(".jsonl") or filename.endswith(".json")):
            filepath = os.path.join(input_dir, filename)
            rewritten_responses = read_file(filepath)
            predictions = []
            for item in tqdm(rewritten_responses, desc="Predicting labels"):
                text = item["generated_text"]
                predicted_label = predict_label(text)
                predictions.append({"id": item["id"], "predicted_label": predicted_label})

            output_dir = f"{output_dir}/{classifier_type}"
            os.makedirs(output_dir, exist_ok=True)

            # Save the predictions
            output_path = '_'.join(filename.split('.')[:-1])
            with open(f"{output_dir}/predicted_label_{output_path}.jsonl", "w") as outfile:
                for prediction in predictions:
                    json.dump(prediction, outfile)
                    outfile.write("\n")

            print(f"Predictions saved to {output_dir}/predicted_label_{output_path}.jsonl")



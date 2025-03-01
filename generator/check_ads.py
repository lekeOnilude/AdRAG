import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm
import random

classifier_model_path = "/home/jmcoelho/11797_Project/models/subtask2_test"
tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def classify_passages(passages):
    inputs = tokenizer(
        passages, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().tolist()


def parse_jsonl_and_classify(file_path, batch_size=300):
    total_answers = 0
    ad_count = 0
    passages = []
    no_ad_answers = []
    ad_answers = []

    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            answer = data["response"]
            passages.append(answer)
            total_answers += 1

            if len(passages) == batch_size:
                predictions = classify_passages(passages)
                for passage, pred in zip(passages, predictions):
                    if pred == 1:
                        ad_answers.append(passage)
                        ad_count += 1
                    else:
                        no_ad_answers.append(passage)
                passages = []

        # Process any remaining passages
        if passages:
            predictions = classify_passages(passages)
            for passage, pred in zip(passages, predictions):
                if pred == 1:
                    ad_answers.append(passage)
                    ad_count += 1
                else:
                    no_ad_answers.append(passage)

    ad_percentage = (ad_count / total_answers) * 100
    return ad_percentage, no_ad_answers, ad_answers


file_path = "/home/jmcoelho/11797_Project/generator/output/marcov2.train/Qwen2.5-0.5B-bidirectional-attn-mntp-marco-passage-hard-negatives-matrioshka-reduction-2/marcov2.train.Qwen2.5-7B-Instruct-10-passage-RAG.jsonl"
ad_percentage, no_ad_answers, ad_answers = parse_jsonl_and_classify(file_path)
print(f"Percentage of answers containing ads: {ad_percentage:.2f}%")

print("\nSample passages with ads:")
for passage in random.sample(ad_answers, min(5, len(ad_answers))):
    print(passage)
    print("############")

# Print some random passages without ads
print("\nSample passages without ads:")
for passage in random.sample(no_ad_answers, min(5, len(no_ad_answers))):
    print(passage)
    print("############")

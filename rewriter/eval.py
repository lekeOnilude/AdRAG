from sklearn.metrics import classification_report
import json

with open("predictions/baseline_text_w_ads_appened.jsonl", "r") as f:
    predictions = []
    for line in f:
        data = json.loads(line)
        predictions.append(data["predicted_label"])

labels = [1] * len(predictions)
report = classification_report(labels, predictions, output_dict=True)
print(report)


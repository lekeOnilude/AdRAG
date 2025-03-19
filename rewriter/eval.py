from sklearn.metrics import classification_report
import json
import os
from tqdm import tqdm



# with open("./predictions/predicted_labelgemma-7b-it_fewshot_0.jsonl", "r") as f:
#     predictions = []
#     for line in f:
#         data = json.loads(line)
#         predictions.append(data["predicted_label"])

# labels = [1] * len(predictions)
# report = classification_report(labels, predictions, output_dict=True)
# print(report)
# print(f"Accuracy: {report['accuracy']}")
# print(f"F1-score: {report['weighted avg']['f1-score']}")


def evaluate_predictions(predictions_file):
    """
    Evaluates predictions from a JSONL file against a ground truth of all 1s.

    Args:
        predictions_file (str): Path to the JSONL file containing predictions.

    Returns:
        dict: A dictionary containing the classification report, accuracy, and F1-score.
    """
    with open(predictions_file, "r") as f:
        predictions = []
        for line in f:
            data = json.loads(line)
            predictions.append(data["predicted_label"])

    labels = [1] * len(predictions)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)  # handle zero division
    return {
        "file": predictions_file.split("/")[-1].split(".")[0],
        "report": report,
        "accuracy": report['accuracy'],
        "f1-score": report['weighted avg']['f1-score'],
    }


def evaluate_all_jsonl_files(directory):
    """
    Evaluates all JSONL files in a directory containing predictions.

    Args:
        directory (str): Path to the directory containing JSONL files.
    """
    results = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            result = evaluate_predictions(filepath)
            if result:
              results.append(result)

    # Print a summary
    for result in results:
      print(f"\nEvaluation for {result['file']}:")
      print(f"  Accuracy: {result['accuracy']}")
      print(f"  F1-score: {result['f1-score']}")



if __name__ == "__main__":
    predictions_directory = "predictions/touche_data/classifier_v1"
    evaluate_all_jsonl_files(predictions_directory)
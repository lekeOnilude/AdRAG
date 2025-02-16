import json
import pandas as pd
from collections import defaultdict

# Load the data
responses_file = "./data/subtask-2/responses-train.jsonl"
labels_file = "./data/subtask-2/responses-train-labels.jsonl"

responses = []
labels = []

with open(responses_file, "r") as f:
    for line in f:
        responses.append(json.loads(line))

with open(labels_file, "r") as f:
    for line in f:
        labels.append(json.loads(line))

# Convert to DataFrame
responses_df = pd.DataFrame(responses)
labels_df = pd.DataFrame(labels)

# Merge the dataframes on 'id'
merged_df = pd.merge(responses_df, labels_df, on="id")


# Function to analyze differences between ad and non-ad responses
def analyze_differences(df):
    grouped = df.groupby("query")
    differences = defaultdict(list)

    for query, group in grouped:
        ad_responses = group[group["label"] == 1]
        non_ad_responses = group[group["label"] == 0]

        if not ad_responses.empty and not non_ad_responses.empty:
            ad_response = ad_responses.iloc[0]
            non_ad_response = non_ad_responses.iloc[0]

            differences[query].append(
                {
                    "ad_response": ad_response["response"],
                    "non_ad_response": non_ad_response["response"],
                    "advertisement": ad_response["advertisement"],
                }
            )

    return differences


# Function to perform statistical analysis
def perform_statistical_analysis(df):
    total_responses = len(df)
    ad_responses = df[df["label"] == 1]
    non_ad_responses = df[df["label"] == 0]

    num_ad_responses = len(ad_responses)
    num_non_ad_responses = len(non_ad_responses)

    ad_percentage = (num_ad_responses / total_responses) * 100
    non_ad_percentage = (num_non_ad_responses / total_responses) * 100

    print(f"Total responses: {total_responses}")
    print(f"Number of ad responses: {num_ad_responses} ({ad_percentage:.2f}%)")
    print(
        f"Number of non-ad responses: {num_non_ad_responses} ({non_ad_percentage:.2f}%)"
    )

    # Additional statistics
    ad_lengths = ad_responses["response"].apply(len)
    non_ad_lengths = non_ad_responses["response"].apply(len)

    print(f"Average length of ad responses: {ad_lengths.mean():.2f} characters")
    print(f"Average length of non-ad responses: {non_ad_lengths.mean():.2f} characters")


# Analyze differences
differences = analyze_differences(merged_df)

# Perform statistical analysis
##perform_statistical_analysis(merged_df)

# Print some examples
for query, diff in list(differences.items())[:5]:
    print(f"Query: {query}")
    for d in diff:
        print(f"######################################")
        print(f"Ad Response: {d['ad_response']}")
        print(f"--------------------------------------")
        print(f"Non-Ad Response: {d['non_ad_response']}")
        print(f"--------------------------------------")
        print(f"Advertisement: {d['advertisement']}")
        print("\n")

import json
import argparse
from collections import defaultdict
import random

random.seed(17121998)


def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analyze_dataset(data):
    """Perform analysis on the dataset."""
    query_count = len(data)
    ad_counts = []
    passage_counts = []
    passage_sizes = []
    ad_description_sizes = []
    query_examples = []

    queries_without_ads = 0

    for entry in data:
        query_examples.append(entry["query"]["text"])

        candidates = entry["candidates"]
        passages_per_query = len(candidates)
        passage_counts.append(passages_per_query)

        for passage in candidates:
            passage_sizes.append(len(passage["doc"]["segment"]))

        advertisements = [x for x in entry["advertisements"] if x]

        if not advertisements:
            queries_without_ads += 1
            continue

        ad_counts.append(len(advertisements))
        for ad in [x for x in entry["advertisements"] if x]:
            ad_description_sizes.append(len(ad["type"]) + len(ad["qualities"]))

    # Compute statistics
    avg_ads_per_query = sum(ad_counts) / (query_count - queries_without_ads)
    avg_ad_desc_size = (
        sum(ad_description_sizes) / len(ad_description_sizes)
        if ad_description_sizes
        else 0
    )
    min_passages = min(passage_counts)
    max_passages = max(passage_counts)
    avg_passages = sum(passage_counts) / query_count
    avg_passage_size = sum(passage_sizes) / len(passage_sizes)

    # Print analysis results
    print(f"Total queries: {query_count}")
    print(f"Example queries: {random.sample(query_examples, 5)}")
    print(f"Queries without ads: {queries_without_ads}")
    print(
        f"For the queries that have ads, average ads per query: {avg_ads_per_query:.2f}"
    )
    print(f"Average ad description size: {avg_ad_desc_size:.2f} characters")
    print(
        f"Passages per query (min/max/avg): {min_passages}/{max_passages}/{avg_passages:.2f}"
    )
    print(f"Average passage size: {avg_passage_size:.2f} characters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Webis Generated Native Ads 2024 dataset"
    )

    data = load_jsonl("./data/subtask-1/touche-task4-2025-segments.jsonl")
    analyze_dataset(data)

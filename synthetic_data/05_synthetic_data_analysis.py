"""
1. Meta topic frequency distribution
    - from origianl (touche) data
    - from structured synthetic data

2. Length of the text comparison by word count frequency distribution
    - word count frequency from the original (touche) data (data/subtask-2/response-<train, validation, test>.jsonl ; 'response' field)
    - word count frequency from the naive synthetic data (synthetic_data/data/single-prompt-multi-model.jsonl; 'without_ad' and 'with_ad' field)
    - word count frequency from the structured synthetic data (synthetic_data/data/synthetic-all.jsonl ; 'response' field)
"""

import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

# Set up nicer plot styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SYNTH_DATA_DIR_PATH = os.path.join(CUR_DIR_PATH, "data")
# Synthetic data
SYNTH_NAIVE_FP = os.path.join(SYNTH_DATA_DIR_PATH, "single-prompt-multi-model.jsonl")
SYNTH_STRUCTURED_FP = os.path.join(SYNTH_DATA_DIR_PATH, "synthetic-all.jsonl")
# Touche data
TOUCHE_DATA_DIR_PATH = os.path.join(os.path.dirname(CUR_DIR_PATH), "data", "subtask-2")
# Output directory for plots
OUTPUT_DIR = os.path.join(CUR_DIR_PATH, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# TODO:
# load touche train, validation, and test data and combine them into one touche data.
#   train, validation, and test data can be accessed by os.path.join(TOUCHE_DATA_DIR_PATH, "responses-<split>.jsonl")
# load naive synthetic data
# load structured synthetic data

# TODO: comparison of meta_topic frequency distribution from two different data sources
# 1) From the touche data, there is a field called 'meta_topic'
# 2) From the synthetic structured data, there is also a field called 'meta_topic'
# Calculate and print count statistics of the meta_topic from the two sources of dataset
# and plot the frequency distributions in one plot with different color with legend indicating which distribution is from which data source.


# TODO: Comparison of word count frequency distribution from three different data sources
# 1) From the touche data, collect text data from a field called 'response'
# 2) From the synthetic naive data, collect text data from both 'without_ad' and 'with_ad' field
# 3) From the synthetic structured data, collect text data from a field called 'response'
# Calculate and print average word count for each data source
# and plot a word count distributions in one plot with different color with legend indicating which distribution is from which data source.

# TODO: Comparison of word count frequency distribution from three different data sources (ad vs non-ad comparisons)
# For touche data, if you look at 'id' field, and if the id value ends with A, then it means the text is advertisement, and if ends with N, it means the text is not an advertisement.
# For structured synthetic data, if you look at the  'id' field, if it ends with 'A-4', the text is ad, and if ends with 'N-H', the text is not ad.
# For naive synthetic data, it is straightforward that which is ad and which is not as you load text from 'with_ad' and 'without_ad' field.
# On top of the current code, I want to additionally compare the word count distribution of ad-text vs non-ad-text.
# For each data source (three in total), create a word count frequency distribution plot.


def load_jsonl(file_path):
    """Load data from a jsonl file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        print(
            f"Successfully loaded {len(data)} entries from {os.path.basename(file_path)}"
        )
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def count_words(text):
    """Count the number of words in a text string."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def plot_distribution(
    data_dict, title, xlabel, ylabel, filename, bins=30, log_scale=False
):
    """Create a distribution plot for the given data."""
    plt.figure(figsize=(12, 8))

    for label, values in data_dict.items():
        if values:
            sns.histplot(values, bins=bins, label=label, alpha=0.6, kde=True)

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if log_scale:
        plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def print_statistics(data_dict, title):
    """Print basic statistics for each dataset."""
    print(f"\n{title}")
    print("=" * 80)

    for label, values in data_dict.items():
        if not values:
            print(f"{label}: No data available")
            continue

        values_array = np.array(values)
        print(f"{label}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {np.mean(values_array):.2f}")
        print(f"  Median: {np.median(values_array):.2f}")
        print(f"  Min: {np.min(values_array)}")
        print(f"  Max: {np.max(values_array)}")
        print(f"  Std Dev: {np.std(values_array):.2f}")
        print("-" * 40)


def is_ad_touche(item_id):
    """Determine if a Touche item is an advertisement based on ID."""
    if not item_id or not isinstance(item_id, str):
        return None
    if item_id.endswith("A"):
        return True
    elif item_id.endswith("N"):
        return False
    return None


def is_ad_structured(item_id):
    """Determine if a Touche item is an advertisement based on ID."""
    if not item_id or not isinstance(item_id, str):
        return None
    if item_id.endswith("A-4"):
        return True
    elif item_id.endswith("N-H"):
        return False
    return None


def main():
    # Load Touche data (train, validation, test)
    touche_data = []
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(TOUCHE_DATA_DIR_PATH, f"responses-{split}.jsonl")
        touche_data.extend(load_jsonl(file_path))

    print(f"Total Touche data entries: {len(touche_data)}")

    # Load synthetic naive data
    synthetic_naive_data = load_jsonl(SYNTH_NAIVE_FP)

    # Load synthetic structured data
    synthetic_structured_data = load_jsonl(SYNTH_STRUCTURED_FP)

    # ========== META TOPIC FREQUENCY DISTRIBUTION ==========
    print("\nAnalyzing meta topic frequency distribution...")

    # Extract meta topics from Touche data
    touche_meta_topics = [
        item.get("meta_topic", "unknown")
        for item in touche_data
        if "meta_topic" in item
    ]
    touche_meta_topic_counts = Counter(touche_meta_topics)

    # Extract meta topics from synthetic structured data
    synth_meta_topics = [
        item.get("meta_topic", "unknown")
        for item in synthetic_structured_data
        if "meta_topic" in item
    ]
    synth_meta_topic_counts = Counter(synth_meta_topics)

    # Print meta topic statistics
    print("\nMeta Topic Distribution in Touche Data:")
    for topic, count in sorted(
        touche_meta_topic_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / len(touche_meta_topics)) * 100
        print(f"{topic}: {count} ({percentage:.2f}%)")

    print("\nMeta Topic Distribution in Synthetic Structured Data:")
    for topic, count in sorted(
        synth_meta_topic_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / len(synth_meta_topics)) * 100
        print(f"{topic}: {count} ({percentage:.2f}%)")

    # Create comparative bar chart for meta topics
    # all_topics = set(touche_meta_topic_counts.keys()).union(set(synth_meta_topic_counts.keys()))
    all_topics = sorted(
        list(
            set(touche_meta_topic_counts.keys()).union(
                set(synth_meta_topic_counts.keys())
            )
        )
    )

    df_meta = pd.DataFrame(
        {
            "Touche": [touche_meta_topic_counts.get(topic, 0) for topic in all_topics],
            "Synthetic": [
                synth_meta_topic_counts.get(topic, 0) for topic in all_topics
            ],
        },
        index=all_topics,
    )

    # Convert to percentage
    df_meta_percent = df_meta.div(df_meta.sum(axis=0), axis=1) * 100

    plt.figure(figsize=(14, 10))
    df_meta_percent.plot(kind="bar", figsize=(14, 10))
    plt.title("Meta Topic Distribution Comparison", fontsize=16)
    plt.xlabel("Meta Topic", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Data Source")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "meta_topic_distribution.png"), dpi=300)
    plt.close()

    # ========== WORD COUNT FREQUENCY DISTRIBUTION ==========
    print("\nAnalyzing word count frequency distribution...")

    # Calculate word counts for each data source
    touche_word_counts = [
        count_words(item.get("response", ""))
        for item in touche_data
        if "response" in item
    ]

    # Combine both without_ad and with_ad fields as one synthetic naive dataset
    naive_word_counts = []
    for item in synthetic_naive_data:
        if "without_ad" in item:
            naive_word_counts.append(count_words(item.get("without_ad", "")))
        if "with_ad" in item:
            naive_word_counts.append(count_words(item.get("with_ad", "")))

    structured_word_counts = [
        count_words(item.get("response", ""))
        for item in synthetic_structured_data
        if "response" in item
    ]

    # Combine word counts for plotting
    word_count_data = {
        "Touche": touche_word_counts,
        "Synthetic Naive": naive_word_counts,
        "Synthetic Structured": structured_word_counts,
    }

    # Print word count statistics
    print_statistics(word_count_data, "Word Count Statistics")

    # Plot word count distributions
    plot_distribution(
        word_count_data,
        "Word Count Distribution Comparison",
        "Word Count",
        "Frequency",
        "word_count_distribution.png",
        bins=50,
    )

    print(f"\nAnalysis complete. Plots saved to {OUTPUT_DIR}")

    # ========== WORD COUNT BY AD VS NON-AD ==========
    print(
        "\nAnalyzing word count distribution for advertisements vs non-advertisements..."
    )

    # Parse Touche data for ad vs non-ad
    touche_ad_word_counts = []
    touche_non_ad_word_counts = []

    for item in touche_data:
        if "id" in item and "response" in item:
            is_ad = is_ad_touche(item["id"])
            word_count = count_words(item["response"])
            if is_ad is True:
                touche_ad_word_counts.append(word_count)
            elif is_ad is False:
                touche_non_ad_word_counts.append(word_count)

    # Parse naive synthetic data for ad vs non-ad
    naive_ad_word_counts = [
        count_words(item.get("with_ad", ""))
        for item in synthetic_naive_data
        if "with_ad" in item
    ]
    naive_non_ad_word_counts = [
        count_words(item.get("without_ad", ""))
        for item in synthetic_naive_data
        if "without_ad" in item
    ]

    # Parse structured synthetic data for ad vs non-ad
    structured_ad_word_counts = []
    structured_non_ad_word_counts = []

    for item in synthetic_structured_data:
        if "id" in item and "response" in item:
            is_ad = is_ad_structured(item["id"])
            word_count = count_words(item["response"])
            if is_ad is True:
                structured_ad_word_counts.append(word_count)
            elif is_ad is False:
                structured_non_ad_word_counts.append(word_count)

    # Print statistics for ad vs non-ad text
    print("\nWord Count Statistics for Advertisement vs Non-Advertisement Text:")
    print("=" * 80)

    ad_non_ad_stats = {
        "Touche - Advertisement": touche_ad_word_counts,
        "Touche - Non-Advertisement": touche_non_ad_word_counts,
        "Synthetic Naive - Advertisement": naive_ad_word_counts,
        "Synthetic Naive - Non-Advertisement": naive_non_ad_word_counts,
        "Synthetic Structured - Advertisement": structured_ad_word_counts,
        "Synthetic Structured - Non-Advertisement": structured_non_ad_word_counts,
    }

    print_statistics(ad_non_ad_stats, "Ad vs Non-Ad Word Count Statistics")

    # Create separate plots for each data source comparing ad vs non-ad
    # 1. Touche Data
    touche_ad_data = {
        "Advertisement": touche_ad_word_counts,
        "Non-Advertisement": touche_non_ad_word_counts,
    }

    plot_distribution(
        touche_ad_data,
        "Touche Data: Word Count Distribution (Ad vs Non-Ad)",
        "Word Count",
        "Frequency",
        "touche_ad_vs_non_ad_word_count.png",
        bins=40,
    )

    # 2. Synthetic Naive Data
    naive_ad_data = {
        "Advertisement": naive_ad_word_counts,
        "Non-Advertisement": naive_non_ad_word_counts,
    }

    plot_distribution(
        naive_ad_data,
        "Synthetic Naive Data: Word Count Distribution (Ad vs Non-Ad)",
        "Word Count",
        "Frequency",
        "naive_ad_vs_non_ad_word_count.png",
        bins=40,
    )

    # 3. Synthetic Structured Data
    structured_ad_data = {
        "Advertisement": structured_ad_word_counts,
        "Non-Advertisement": structured_non_ad_word_counts,
    }

    plot_distribution(
        structured_ad_data,
        "Synthetic Structured Data: Word Count Distribution (Ad vs Non-Ad)",
        "Word Count",
        "Frequency",
        "structured_ad_vs_non_ad_word_count.png",
        bins=40,
    )

    # Combined plot with all ad vs non-ad distributions
    plt.figure(figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    color_idx = 0

    for label, values in ad_non_ad_stats.items():
        if values:
            sns.kdeplot(values, label=label, color=colors[color_idx])
            color_idx += 1

    plt.title(
        "Word Count Distribution: Advertisement vs Non-Advertisement Across All Sources",
        fontsize=16,
    )
    plt.xlabel("Word Count", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "combined_ad_vs_non_ad_word_count.png"), dpi=300
    )
    plt.close()


if __name__ == "__main__":
    main()

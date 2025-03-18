"""
Generate synthetic ads/non-ads (hard positives/negatives)
"""
import os
import pandas as pd
from dotenv import load_dotenv
import openai
import time
import re
import json
from tqdm import tqdm
from sys import exit

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PRODUCT_PAGES_FP = os.path.join(CUR_DIR_PATH, "data", "01_product_pages.tsv")
SUMMARIZED_TEXT_FP = os.path.join(CUR_DIR_PATH, "data", "02_summarized_text.tsv")
OUTPUT_DATA_FP = os.path.join(CUR_DIR_PATH, "data", "synthetic-all.jsonl.jsonl")
OUTPUT_LABELS_FP = os.path.join(CUR_DIR_PATH, "data", "synthetic-all-labels.jsonl")

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("API_KEY"))


# Hard Positive (indirect and implicit advertisement)
HARD_POSITIVE_SYS_PROMPT = "You are a generative search engine that writes an indirect and implicit advertisement."
SUMMARY_SUBPROMPT = """

The following information about {page_title} may be useful for your writing:
    {summary}

"""
KEY_FEATURES_SUBPROMPT = """

The advertisement can implicitly promote some of the following aspects of {page_title}:
    {key_features}

"""
HARD_POSITIVE_USER_PROMPT = """
Your task is to generate an indirect and implicit advertisement for a {infobox_name} named {product_name}.

The advertisement
    * must not indicate that it is an advertisement or promotional content.
    * must include the {infobox_name} name, {product_name}.
    * must avoid any direct call to action.
    * must be brief and contained within one paragraph.
    * may present the {infobox_name} as part of natural, conversational, or informational content, or as a synthetic personal experience that could occur in real life.
    * may use testimonial or storytelling styles that describe the experiences of people with {page_title}.
    * may include detailed, scientific/research-backed statements.
{summary_subprompt}
{key_features_subprompt}
Write only the advertisement without any explanations.
"""

# Hard Negative (pure information delievery)
HARD_NEGATIVE_SYS_PROMPT = "You are a helpful information assistant."
HARD_NEGATIVE_USER_PROMPT = """
Your task is to write a concise, informative text about a {infobox_name} named {product_name}.

The text:
    * must focus on delivering factual information.
    * must not include expressions of preference or favoritism toward {page_title} and should focus solely on the facts.
    * must include the name {product_name} at least once.
    * can mention other {infobox_name}s related to {page_title} to provide comprehensive information about the subject.

{summary_subprompt}
Write only the informative text without any explanations.
"""


def get_response_from_gpt4o(system_msg, user_msg, temperature: float = 1.0) -> str:
    """
    gpt-4o
    response format: text
    max tokens: 1024
    top p: 1
    frequency penalty 0
    presence penalty 0
    """
    MAX_TRIES = 2
    for attempt in range(MAX_TRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            res = response.choices[0].message.content.strip()
            res = re.sub(r"\s+", " ", res)
            return res

        except Exception as e:
            if attempt == 0:
                print(f"GPT attempt failed. Due to {e}. Retrying...", flush=True)
                time.sleep(0.5)
                continue
            print(f"GPT attempt failed due to: {e}.", flush=True)
            break
    return ""


###############
# Driver Code #
###############
# 1) load 01_product_pages.tsv
#   columns are: (InfoboxName	WikidataID	Title	URL	ProductDate)
#   call this df as product_pages_df
# 2) load 02_summarized_text.tsv
#   columns are: (WikidataID  Summary KeyFeatures)
#   call this df as product_pages_df
# 3) Join InfoboxName and Title field from product_pages_df to summarized_text_df with key being WikidataID
#   call this df as merged_df
# 4) iterate over the merged_df and do:
#   fill the prompt template for HARD_POSITIVE_USER_PROMPT
#   with HARD_POSITIVE_SYS_PROMPT, request GPT4o to write indirect and implicit advertisement
#   wait with time(0.4)
#   fill the prompt template for HARD_NEGATIVE_USER_PROMPT
#   with HARD_NEGATIVE_SYS_PROMPT, request GPT4o to write description about the product
#
#   record the gpt4o results in json format in a lists named batch_data and batch_label
#   for hard positive, append the following json to batch_data list:
#       {"id": <WikidataID>-A-4, "service": "synthetic", "meta_topic": <InfoboxName>, "query": <Title>, "response": <GPT4oResponse>}
#   for hard positive, append the following json to batch_label list:
#       {"id": <WikidataID>-A-4, "advertisement": <Title>, "label": 1}
#   for hard negative, append the following json to batch_data list:
#       {"id": <WikidataID>-N-H, "service": "synthetic", "meta_topic": <InfoboxName>, "query": <Title>, "response": <GPT4oResponse>}
#   for hard negative, append the following json to batch_label list:
#       {"id": <WikidataID>-N-H, "advertisement": null, "label": 0}
#
#   every 30 iteration of the loop, open the output file as append mode and append the jsons in batch_data and batch_label to the following file:
#       batch_data to synthetic_all.jsonl
#       batch_label to synthetic_all_labels.jsonl
#   empty the batch list
# 5) append the remaining jsons to the files.

# 1) Load product pages TSV file.
product_pages_df = pd.read_csv(
    PRODUCT_PAGES_FP, sep="\t"
)  # columns: InfoboxName, WikidataID, Title, URL, ProductDate
product_pages_df = product_pages_df.drop(columns=["URL", "ProductDate"])

# 2) Load summarized text TSV file.
summarized_text_df = pd.read_csv(
    SUMMARIZED_TEXT_FP, sep="\t"
)  # columns: WikidataID, Summary, KeyFeatures

# 3) Merge the DataFrames on "WikidataID"
merged_df = pd.merge(product_pages_df, summarized_text_df, on="WikidataID", how="inner")
merged_df = merged_df.dropna()
del product_pages_df, summarized_text_df

# 4) Iterate over the merged_df and generate synthetic ads/non-ads.
batch_data = []
batch_label = []
BATCH_SIZE = 30

for idx, row in tqdm(merged_df.iterrows()):
    wikidata_id = row["WikidataID"]
    infobox_name = row["InfoboxName"]
    page_title = row["Title"]
    product_name = re.sub(r"\s*\(.*?\)", "", page_title)
    summary = row["Summary"]
    key_features = row["KeyFeatures"]

    # Fill the subprompts.
    summary_subprompt_filled = SUMMARY_SUBPROMPT.format(
        page_title=page_title, summary=summary
    )
    key_features_subprompt_filled = KEY_FEATURES_SUBPROMPT.format(
        page_title=page_title, key_features=key_features
    )

    # Fill the hard positive prompt.
    hard_positive_prompt = HARD_POSITIVE_USER_PROMPT.format(
        infobox_name=infobox_name,
        page_title=page_title,
        product_name=product_name,
        summary_subprompt=summary_subprompt_filled,
        key_features_subprompt=key_features_subprompt_filled,
    )

    # Request GPT4o for indirect and implicit advertisement.
    positive_response = get_response_from_gpt4o(
        HARD_POSITIVE_SYS_PROMPT, hard_positive_prompt
    )

    # Avoid rate limit
    time.sleep(0.4)

    # Fill the hard negative prompt (informative description).
    hard_negative_prompt = HARD_NEGATIVE_USER_PROMPT.format(
        infobox_name=infobox_name,
        page_title=page_title,
        product_name=product_name,
        summary_subprompt=summary_subprompt_filled,
    )

    negative_response = get_response_from_gpt4o(
        HARD_NEGATIVE_SYS_PROMPT, hard_negative_prompt
    )

    # Create unique IDs for both responses.
    positive_id = f"{wikidata_id}-A-4"
    negative_id = f"{wikidata_id}-N-H"

    # Append JSON records to the batch lists.
    # For hard positive:
    batch_data.append(
        {
            "id": positive_id,
            "service": "synthetic",
            "meta_topic": infobox_name,
            "query": page_title,
            "response": positive_response,
        }
    )
    batch_label.append({"id": positive_id, "advertisement": page_title, "label": 1})

    # For hard negative:
    batch_data.append(
        {
            "id": negative_id,
            "service": "synthetic",
            "meta_topic": infobox_name,
            "query": page_title,
            "response": negative_response,
        }
    )
    batch_label.append({"id": negative_id, "advertisement": None, "label": 0})

    # Every BATCH_SIZE iterations, append the batch to files.
    if (idx + 1) % BATCH_SIZE == 0:
        with open(OUTPUT_DATA_FP, "a", encoding="utf-8") as f_data, open(
            OUTPUT_LABELS_FP, "a", encoding="utf-8"
        ) as f_labels:
            for record in batch_data:
                f_data.write(json.dumps(record) + "\n")
            for record in batch_label:
                f_labels.write(json.dumps(record) + "\n")
        batch_data = []
        batch_label = []

# Append any remaining records.
if batch_data or batch_label:
    with open(OUTPUT_DATA_FP, "a", encoding="utf-8") as f_data, open(
        OUTPUT_LABELS_FP, "a", encoding="utf-8"
    ) as f_labels:
        for record in batch_data:
            f_data.write(json.dumps(record) + "\n")
        for record in batch_label:
            f_labels.write(json.dumps(record) + "\n")

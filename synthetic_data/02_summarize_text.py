"""
1) Fetch Wikipedia page text
2) Summarize product description focusing on the quality of the product to be advertised

Saves a data file as follows:
wikidataID, summarized_text, quality
"""

import requests
from dotenv import load_dotenv
import openai
import os
import pandas as pd
import re
import tiktoken
from tqdm import tqdm
import time

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_FP = os.path.join(CUR_DIR_PATH, "data", "02_summarized_text.tsv")


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("API_KEY"))

HEADERS = {"User-Agent": "toeunk@cs.cmu.edu"}

SYSTEM_MESSAGE = "You are a text summarization expert."
PROMPT_BASIS = """
Summarize the following Wikipeida page about {page_title} into a few paragraphs, and list a few key features and quality to advertise.
For summary, focus on the most relevant information for creating an advertisement text about the {infobox_name}. Extract key features, unique selling points, and any historical or factual details that would make the {infobox_name} more appealing to potential consumers. Keep the summary concise and suitable for marketing purposes. Avoid excessive technical details unless they enhance the product's appeal.

Format the output as follows:
<summary> [insert the summary here] </summary>
<keyfeatures> [list key features and quality to advertise separated by ;] </keyfeatures> 

Wikipedia page about {page_title}:
{page_body}
"""


def load_product_pages(fp) -> pd.DataFrame:
    return pd.read_csv(fp, sep="\t")


def get_infobox_wikitext(page_title):
    """Retrieve the raw wikitext of a Wikipedia page."""
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "rvslots": "main",
    }
    response = requests.get(URL, params=params, headers=HEADERS)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return (
            page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")
        )
    return ""


def truncate_text_to_max_tokens(text, model_name="gpt-4o", max_tokens=2048):
    """
    Truncate the input text to fit within the token limit while keeping meaning intact.
    """
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)

    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        text = enc.decode(truncated_tokens)  # Convert tokens back to text

    return text


def get_summary_and_key_features_from_gpt4o(
    system_msg, user_msg, temperature: float = 0.5
) -> str:
    """
    gpt-4o
    response format: text
    max tokens: 2048
    top p: 1
    frequency penalty 0
    presence penalty 0
    """
    MAX_TRIES = 2
    for attempt in range(MAX_TRIES):
        time.sleep(1)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            res = response.choices[0].message.content.strip()
            # Extract summary and key features text from tags
            summary_match = re.search(r"<summary>(.*?)</summary>", res, re.DOTALL)
            keyfeatures_match = re.search(
                r"<keyfeatures>(.*?)</keyfeatures>", res, re.DOTALL
            )

            summary = summary_match.group(1).strip() if summary_match else ""
            key_features = (
                keyfeatures_match.group(1).strip() if keyfeatures_match else ""
            )

            return summary, key_features

        except openai.error.InvalidRequestError as e:
            # Check if the error is specifically due to input length exceeding the model's limit
            if attempt == 0:
                print("Input exceeded the context limit. Truncating and retrying...")
                # Truncate system and user messages to fit within max_input_tokens
                user_msg = truncate_text_to_max_tokens(
                    user_msg, model_name="gpt-4o", max_tokens=8000
                )
                continue  # Retry with truncated input
            print(f"Summarization attempt failed due to: {e}.")
            break

        except Exception as e:
            if attempt == 0:
                print(f"Summarization attempt failed. Due to {e}. Retrying...")
                continue
            print(f"Summarization attempt failed due to: {e}.")
            break
    return "", ""


###############
# Driver Code #
###############
# 0) Iterate over product_page_df
# 1) Fetch Wikipedia page text by get_infobox_wikitext() function
# 2) Prompt GPT4o and get summarized text and key features. When prompting, fill the prompt template with title, infobox_name and wikipedia text
# 3) Save a TSV file (OUTPUT_FP) with the following columns:
# WikidataID    Summary    KeyFeatures

product_page_df = load_product_pages(
    os.path.join(CUR_DIR_PATH, "data", "01_product_pages.tsv")
)
product_page_df = product_page_df.drop(columns=["URL", "ProductDate"])

# Set batch size
BATCH_SIZE = 50
batch = []

# Check if the output file exists; if not, create it with header
if not os.path.exists(OUTPUT_FP):
    with open(OUTPUT_FP, "w", encoding="utf-8") as f:
        f.write("WikidataID\tSummary\tKeyFeatures\n")

for idx, row in tqdm(product_page_df.iterrows()):
    wikidata_id = row["WikidataID"]
    page_title = row["Title"]
    infobox_name = row["InfoboxName"]

    # 1) Fetch Wikipedia page text
    page_body = get_infobox_wikitext(page_title)

    # 2) Create the prompt by filling the template
    prompt = PROMPT_BASIS.format(
        page_title=page_title, infobox_name=infobox_name, page_body=page_body
    )

    # 3) Prompt GPT-4o and get summarized text and key features
    summary, key_features = get_summary_and_key_features_from_gpt4o(
        SYSTEM_MESSAGE, prompt
    )

    # Append the result for this page
    batch.append(
        {"WikidataID": wikidata_id, "Summary": summary, "KeyFeatures": key_features}
    )
    # Every BATCH_SIZE rows, append the batch to the TSV file
    if len(batch) >= BATCH_SIZE:
        df_batch = pd.DataFrame(batch)
        # Append without writing header
        df_batch.to_csv(OUTPUT_FP, sep="\t", index=False, header=False, mode="a")
        batch = []  # Reset the batch

# If there are any remaining rows, append them
if batch:
    df_batch = pd.DataFrame(batch)
    df_batch.to_csv(OUTPUT_FP, sep="\t", index=False, header=False, mode="a")

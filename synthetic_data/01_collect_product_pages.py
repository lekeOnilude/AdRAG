"""
Saves a data file as follows:
Infobox Name, wikidataID, page title, page url, is_product
"""
import requests
import datetime
import pandas as pd
import random
import os
from tqdm import tqdm
import mwparserfromhell
import re

random.seed(42)

# List of Wikidata properties that, if found, indicate a product page.
"""
P50 - author
main creator(s) of a written work (use on works, not humans)

P86 - composer
person(s) who wrote the music

P110 - illustrator
person drawing the pictures or taking the photographs in a book or similar work

P123 – publisher
Typically used for media products, books, or software where a publishing house is involved.

P162 - producer
person(s) who produced the film, musical work, theatrical production, etc.

P170 - creator
maker of this creative work or other object (where no more specific property exists)

P176 – manufacturer
Identifies the company or entity that manufactured the product. This is the primary property for capturing brand information, although it may be absent on some pages.

P178 – developer
Often used for software, video games, or products that are “developed” rather than manufactured.

P179 – product series
Indicates if the product is part of a series or family of related products.

P287 - designed by
person or organization which designed the object

P593 – model number
Captures the specific model identifier for the product.

P676 - lyricist
author of song lyrics

P943 - programmer
the programmer that wrote the piece of software

P3640 - National Drug Code
pharmaceutical code issued by the Food and Drug Administration for every drug product

P4087 - MyAnimeList manga ID
MyAnimeList manga ID

P8731 - AniList manga ID
identifier for AniList.co manga and light novels

P9618- AlternativeTo software ID
identifier used in the crowdsourced software recommendation website AlternativeTo.net

P9897 - App Store age rating
content rating category by Apple for software published at the App Store

P12969 - game designer
person(s) who devised and developed this game
"""
PRODUCT_PROPERTIES = [
    "P50",
    "P86",
    "P110",
    "P123",
    "P162",
    "P170",
    "P176",
    "P178",
    "P179",
    "P123",
    "P287",
    "P593",
    "P676",
    "P943",
    "P3640",
    "P4087",
    "P8731",
    "P9618",
    "P9897",
    "P12969",
]

DATE_PROPERTIES = ["P571", "P577"]

HEADERS = {"User-Agent": "YourAppName/1.0 (your.email@example.com)"}


def get_pages_using_infobox(template_name, namespace=0) -> list[dict]:
    """
    Retrieve pages that embed a given infobox template.

    Args:
        template_name (str): The exact template title, e.g. "Template:Infobox_software"
        namespace (int): Wikipedia namespace (0 for main articles)

    Returns:
        list: A list of dictionaries, each containing page info (e.g. page id and title)
    """
    URL = "https://en.wikipedia.org/w/api.php"
    pages = []
    params = {
        "action": "query",
        "list": "embeddedin",
        "eititle": template_name,
        "einamespace": namespace,
        "format": "json",
        "eilimit": "max",  # maximum items per request
    }

    while True:
        response = requests.get(URL, params=params, headers=HEADERS)
        data = response.json()
        pages.extend(data["query"]["embeddedin"])

        # Check if there is a continuation token to get the next batch
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return pages


def sample_pages(pages: list[dict], k=300) -> list[dict]:
    if len(pages) > k:
        random.choices()
        return random.sample(pages, k=k)
    return pages


def get_wikidata_id(page_title):
    """Fetch the Wikidata ID for a given Wikipedia page."""
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "pageprops",
        "format": "json",
    }
    response = requests.get(URL, params=params, headers=HEADERS)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("pageprops", {}).get("wikibase_item")
    return None


def get_wikidata_claims(wikidata_id):
    """Fetch Wikidata claims for the given Wikidata ID."""
    URL = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "props": "claims",
    }
    response = requests.get(URL, params=params, headers=HEADERS)
    data = response.json()
    entity = data.get("entities", {}).get(wikidata_id, {})
    return entity.get("claims", {})


def is_product_from_wikidata_prop(claims):
    """
    Check if any of the candidate properties is present in the Wikidata claims.
    Returns True if at least one candidate property is found.
    """
    for prop in PRODUCT_PROPERTIES:
        if prop in claims and claims[prop]:
            # Found at least one candidate property.
            return True
    return False


def is_product_page(page_title, return_with_wikidata_info=False):
    """
    Determines if a given Wikipedia page describes a specific product.
    Returns:
        - (True, wikidata_id, claims) if a product page
        - (False, wikidata_id, claims) if not a product page
    """
    wikidata_id = get_wikidata_id(page_title)
    if not wikidata_id:
        return (False, None, None) if return_with_wikidata_info else False

    claims = get_wikidata_claims(wikidata_id)
    if is_product_from_wikidata_prop(claims):
        return (True, wikidata_id, claims) if return_with_wikidata_info else True

    return (False, wikidata_id, claims) if return_with_wikidata_info else False


def extract_year_from_claim(claims, prop):
    """
    Extracts the year from a given Wikidata property claim.
    Returns a date object set to January 1st of the extracted year.
    If the claim is not found or parsing fails, returns None.
    """
    if prop not in claims:
        return None
    for claim in claims[prop]:
        try:
            time_str = claim["mainsnak"]["datavalue"]["value"][
                "time"
            ]  # e.g., "+2012-03-15T00:00:00Z"
            year_str = time_str[1:5]  # Extract the year (e.g., "2012")
            year_int = int(year_str)
            return datetime.date(year_int, 1, 1)
        except (KeyError, ValueError):
            continue
    return None


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


def parse_year_from_infobox(page_title):
    """
    Fallback: Retrieves the raw wikitext of the given page and uses
    mwparserfromhell to search the infobox for fields like 'inception' or 'released'.
    It then uses a regex to capture a 4-digit year (e.g., 1999 or 2012) past 1940 and
    returns a date object set to January 1st of that year.
    """
    wikitext = get_infobox_wikitext(
        page_title
    )  # Assumes this function is defined elsewhere.
    if not wikitext:
        return None

    wikicode = mwparserfromhell.parse(wikitext)
    for template in wikicode.filter_templates():
        if "infobox" in template.name.lower():
            # Try common fields for product date.
            for field in ["inception", "released"]:
                if template.has(field):
                    val = template.get(field).value.strip_code().strip()
                    # Use regex to extract a 4-digit year but only year from 1940
                    match = re.search(r"\b(19[4-9]\d|20\d{2})\b", val)
                    if match:
                        try:
                            year_int = int(match.group(0))
                            return datetime.date(year_int, 1, 1)
                        except ValueError:
                            continue
    return None


def get_product_date(claims, page_title=None):
    """
    Attempts to retrieve a product's date from candidate date properties
    (P571 and P577) by extracting only the year information.
    If multiple dates are found, returns the most recent.
    If no date is found and page_title is provided, falls back to parsing the infobox.
    """
    date_values = []
    for prop in DATE_PROPERTIES:
        date_val = extract_year_from_claim(claims, prop)
        if date_val:
            date_values.append(date_val)
    if date_values:
        # Return the most recent date (largest value).
        return max(date_values)
    # Fallback to parsing the infobox if page_title is provided.
    if page_title:
        return parse_year_from_infobox(page_title)
    return None


###############
# Driver Code #
###############
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
os.makedirs(os.path.join(CUR_DIR_PATH, "data"), exist_ok=True)
OUTPUT_FP = os.path.join(CUR_DIR_PATH, "data", "01_product_pages.tsv")
FILE_EXISTS = os.path.exists(
    OUTPUT_FP
)  # Check if file exists so we know whether to write header.
PER_INFOBOX_MAX_PAGE_COUNT = 300


# Hand-Selected Infobox name spaces that are likely to contain pages related to a single product that can be advertised.
# List of all infoboxes can be found at: https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes
infobox_names = [
    "product",
    "brand",
    "automobile",
    "motorcycle",
    "tractor",
    "calculator",
    "computing device",
    "keyboard",
    "software",
    "camera",
    "mobile phone",
    "night vision device",
    "synthesizer",
    "tool",
    "watch",
    "pinball",
    "toy",
    "film",
    "book",
    "Asian comic series",
    "comic",
    "musical",
    "furniture",
    "video game",
    "drug",
]


for infobox_name in tqdm(infobox_names):
    # Get template with infobox name
    template = f"Template:Infobox {infobox_name}"
    template = template.replace(" ", "_")

    pages = get_pages_using_infobox(template)
    random.shuffle(
        pages
    )  # shuffle pages so that they are not listed in alphabetical order

    # Collect page info
    page_info_dicts: list[dict] = []
    for page in pages:
        page_title = page["title"]
        product_page_flag, wikidata_id, claims = is_product_page(
            page_title, return_with_wikidata_info=True
        )
        # collect additional info if page is about specific product
        if product_page_flag:
            # 1. page url
            page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            # 2. product release date
            product_date = get_product_date(claims)
            page_info_dicts.append(
                {
                    "InfoboxName": infobox_name,
                    "WikidataID": wikidata_id,
                    "Title": page_title,
                    "URL": page_url,
                    "ProductDate": product_date,
                }
            )

    # sort pages based on product recency
    sorted_page_info_dicts: list[dict] = sorted(
        page_info_dicts,
        key=lambda x: x["ProductDate"]
        if x["ProductDate"] is not None
        else datetime.date.min,
        reverse=True,
    )
    # limit per-infobox-page-count to 300
    sorted_page_info_dicts = sorted_page_info_dicts[:PER_INFOBOX_MAX_PAGE_COUNT]

    # Convert to DataFrame and append to TSV file.
    if sorted_page_info_dicts:
        df = pd.DataFrame(
            sorted_page_info_dicts,
            columns=["InfoboxName", "WikidataID", "Title", "URL", "ProductDate"],
        )
        df.to_csv(OUTPUT_FP, sep="\t", mode="a", header=not FILE_EXISTS, index=False)
        # Once the file is created, ensure that header isn't written again.
        FILE_EXISTS = True

"""
Saves a data file as follows:
Infobox Name, wikidataID, page title, page url, is_product
"""
import requests

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
CANDIDATE_PROPERTIES = [
    "P50",
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
    "P943",
    "P3640",
    "P4087",
    "P8731",
    "P9618",
    "P9897", 
    "P12969",
]

HEADERS = {"User-Agent": "YourAppName/1.0 (your.email@example.com)"}


def get_pages_using_infobox(template_name, namespace=0):
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
    for prop in CANDIDATE_PROPERTIES:
        if prop in claims and claims[prop]:
            # Found at least one candidate property.
            return True
    return False


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


# def parse_infobox_for_brand(wikitext):
#     """
#     Parses the infobox from the wikitext to see if it contains keys
#     that may indicate product/brand information.
#     """
#     wikicode = mwparserfromhell.parse(wikitext)
#     for template in wikicode.filter_templates():
#         if "infobox" in template.name.lower():
#             for field in ["manufacturer", "publisher", "brand", "made by", "product"]:
#                 if template.has(field):
#                     value = template.get(field).value.strip_code().strip()
#                     if value:
#                         return value
#     return None


def is_product_page(page_title):
    """
    Determines if a given Wikipedia page describes a specific product.
    """
    wikidata_id = get_wikidata_id(page_title)
    if wikidata_id:
        claims = get_wikidata_claims(wikidata_id)
        if is_product_from_wikidata_prop(claims):
            return True
    return False

###############
# Driver Code #
###############
# Hand-Selected Infobox name spaces that are likely to contain pages related to a single product that can be advertised.
# List of all infoboxes can be found at: https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes
infobox_names = [
    "film",
    "card game",
    "paintball marker",
    "pinball",
    "toy",
    "book",
    "Asian comic series",
    "comic",
    "musical",
    "furniture",
    "video game",
    "drug",
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
]

for infobox_name in infobox_names:
    # Get template with infobox name
    template = f"Template:Infobox {infobox_name}"
    template = template.replace(" ", "_")

    pages = get_pages_using_infobox(template)
    # TODO: randomly pick N number of pages with random seed to cap the total number per infobox

    # Print the titles and URLs for each page
    # TODO: also collect page text
    page_titles = []
    page_urls = []
    for page in pages:
        page_title = page["title"]
        page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        page_titles.append(page_title)
        page_urls.append(page_url)
        print(f"{page_title} -> {page_url}")

    for title in page_titles:
        product_flag = is_product_page(title)
        print(f"{title}: {'Product page' if product_flag else 'Not a product page'}")

    # TODO
    # Save to file as:
    # Infobox Name, wikidataID, page title, page url, is_product


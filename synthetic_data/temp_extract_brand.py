import requests
import mwparserfromhell

# List of Wikidata properties that, if found, indicate a product page.
# Note: P31 ("instance of") is present on many pages, so you might refine the logic
# by checking for specific values later if needed.
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

# ALLOWED_INSTANCE_QIDS = {"Q571", "Q11424", "Q7397", "Q7889", "Q1344"}

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


def is_product_from_wikidata(claims):
    """
    Check if any of the candidate properties is present in the Wikidata claims.
    Returns True if at least one candidate property is found.
    """
    for prop in CANDIDATE_PROPERTIES:
        if prop in claims and claims[prop]:
            # Found at least one candidate property.
            return True

    # # Check the 'instance of' (P31) property.
    # if "P31" in claims:
    #     for claim in claims["P31"]:
    #         try:
    #             value_id = claim["mainsnak"]["datavalue"]["value"]["id"]
    #             if value_id in ALLOWED_INSTANCE_QIDS:
    #                 return True
    #         except (KeyError, TypeError):
    #             continue
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


def parse_infobox_for_brand(wikitext):
    """
    Parses the infobox from the wikitext to see if it contains keys
    that may indicate product/brand information.
    """
    wikicode = mwparserfromhell.parse(wikitext)
    for template in wikicode.filter_templates():
        if "infobox" in template.name.lower():
            for field in ["manufacturer", "publisher", "brand", "made by", "product"]:
                if template.has(field):
                    value = template.get(field).value.strip_code().strip()
                    if value:
                        return value
    return None


def is_product_page(page_title):
    """
    Determines if a given Wikipedia page describes a specific product.
    First, it attempts to use Wikidata structured data; if that isn't conclusive,
    it falls back to parsing the page's infobox.
    """
    wikidata_id = get_wikidata_id(page_title)
    print(wikidata_id)
    if wikidata_id:
        claims = get_wikidata_claims(wikidata_id)
        print(claims.keys())
        # print(set(claims.keys()).intersection(set(CANDIDATE_PROPERTIES)))
        if is_product_from_wikidata(claims):
            return True
    else:
        print("no wikidata ID")

    # # Fallback: Check the infobox for brand/manufacturer info.
    # wikitext = get_infobox_wikitext(page_title)
    # if wikitext:
    #     brand_info = parse_infobox_for_brand(wikitext)
    #     if brand_info:
    #         return True

    return False


###############
# Driver Code #
###############
page_titles = [
    "Dolby Vision",
    # Likely a specific product
    # "Lego",
    # "Pin-Bot",
    # "Tippmann TPX",
    # "G.I. Joe",
    # "The Simpsons (pinball)",
    # "Vegemite",
    # "Band-Aid",
    # "Nutella",
    # "Lucky Charms",
    # "LGB (trains)",
    # "Porsche 912",
    # "AMC Gremlin",
    # "Mini",
    # "Honda ST series",
    # "Ducati 916",
    # "HP-65",
    # "TI-81",
    # "Nikon D70",
    # "Minolta Maxxum 7000",
    # "Tylenol (brand)",
    # "Cap'n Crunch",
    # "Barbie",
    # "Shenwuji",
    # "Cyber Weapon Z",
    # "I'd_Rather_Be_Right",
    # "The Great Gatsby",
    # "Grand Prix (chair)",
    # "Duck Hunt",
    # "Six Flags Hurricane Harbor Splashtown",
    # "Sony Ericsson P900",
    # "Samsung SPH-i550",
    # "AN/PVS-4",
    # "1PN58",
    # "Yamaha DX7",
    # "Elektron SidStation",
    # "Rolex Datejust",
    # "Omega Seamaster Omegamatic",
    # "HP-01",
    # "Indiana Jones Adventure",
    # "A Wizard of Earthsea",
    # "A Fire Upon the Deep",
    # "Crash (Ballard novel)",
    # "Carmilla",
    # "Blade Runner 2: The Edge of Human",
    # "Back to the Klondike",
    # "Pluto Saves the Ship",
    # "Sheriff of Bullet Valley"
    # # Likely a general concept
    # "Yo-yo",
    # "Doll",
    # "Action figure",
    # "Model car",
    # "Caffeine",
    # "Ketamine",
    # "Amoxicillin",
    # "Toaster",
    # "Apple Inc.",
    # "Diving helmet",
    # "Waterproof wristlet watch"
]

for title in page_titles:
    product_flag = is_product_page(title)
    print(f"{title}: {'Product page' if product_flag else 'Not a product page'}")

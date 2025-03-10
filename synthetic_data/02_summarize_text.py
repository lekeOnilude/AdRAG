"""
1) Fetch Wikipedia page
2) Summarize product description focusing on the quality of the product to be advertised

Saves a data file as follows:
wikidataID, summarized_text, quality
"""
import requests

HEADERS = {"User-Agent": "YourAppName/1.0 (your.email@example.com)"}


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

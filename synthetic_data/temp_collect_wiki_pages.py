import requests


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

    # Define a custom user agent header as recommended by Wikipedia's API guidelines.
    headers = {"User-Agent": "toeunk@andrew.cmu.edu"}

    while True:
        response = requests.get(URL, params=params, headers=headers)
        data = response.json()
        pages.extend(data["query"]["embeddedin"])

        # Check if there is a continuation token to get the next batch
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return pages


# Example usage for the "Infobox furniture" template
template = "Template:Infobox comic"
template = template.replace(" ", "_")
pages = get_pages_using_infobox(template)

# Print the titles and URLs for each page
for page in pages[:20]:
    title = page["title"]
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    print(f"{title} -> {page_url}")

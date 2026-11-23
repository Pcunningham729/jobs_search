"""Small helper functions shared by every fetcher.

Anything two or more fetchers need lives here, so it's written (and fixed)
in one place instead of being copied into each fetcher.
"""

# BeautifulSoup is a library for reading HTML. We use it to strip out tags.
from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """Turn HTML into plain text.

    Example:
        "<p>Build <b>pipelines</b></p><p>Ship it</p>"  ->  "Build\\npipelines\\nShip it"
    """
    # Let BeautifulSoup read the HTML and give back only the text.
    # `separator="\n"` puts a line break where each tag was, so words from
    # different paragraphs don't get glued together.
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # Tidy up: remove spaces at the start and end of each line, and drop
    # lines that are empty.
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:  # an empty string counts as False, so this skips blank lines
            clean_lines.append(line)
    return "\n".join(clean_lines)


def build_location(places: list[str], is_remote: bool) -> str:
    """Combine a job's places into one location string, like "Remote; Minneapolis, MN".

    Lever and Ashby say whether a job is remote in a separate field, not in
    the location text. But the location filter only reads the text. So if a
    job is remote and none of its places already says "remote", we add
    "Remote" at the front.

    Examples:
        build_location(["New York, NY", "Austin, TX"], is_remote=True)
            -> "Remote; New York, NY; Austin, TX"
        build_location(["Remote (US)"], is_remote=True)
            -> "Remote (US)"            (already says remote, nothing added)
        build_location(["Berlin"], is_remote=False)
            -> "Berlin"
    """
    # Clean the list: strip spaces, skip empty places and duplicates.
    clean_places = []
    for place in places:
        place = place.strip()
        if place and place not in clean_places:
            clean_places.append(place)

    # Does any place already mention "remote"? `.lower()` makes the check
    # ignore upper/lower case, so "Remote (US)" counts.
    already_says_remote = False
    for place in clean_places:
        if "remote" in place.lower():
            already_says_remote = True

    if is_remote and not already_says_remote:
        # Put "Remote" first in the list.
        clean_places.insert(0, "Remote")

    return "; ".join(clean_places)

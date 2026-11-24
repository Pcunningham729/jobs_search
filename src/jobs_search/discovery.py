"""Find companies that post jobs on the supported job boards, automatically.

The job-board APIs can't "search all jobs". They only answer "what jobs does
THIS company have?", so the program needs company names. In these APIs a
company is named by a short SLUG (a lowercase, URL-friendly name), like
"gitlab" in boards-api.greenhouse.io/v1/boards/gitlab/jobs.

This module builds that list for you, so you never maintain it:

    1. Collect CANDIDATES (slugs that might be real boards) from:
       - Common Crawl: a free public archive of the web. Its index API lists
         every address it has seen, e.g. every "jobs.lever.co/<slug>/..." page.
       - company_seeds.txt: a starter list of well-known boards, in git.
    2. CHECK each new candidate against the real API: a 200 answer means the
       board exists; 404 means it doesn't, and it's dropped.
    3. Save the boards that exist to companies.csv (in git: it's not personal).

Run it with:  uv run jobs-search discover
"""

import csv
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel

from jobs_search.models import Source

TIMEOUT_SECONDS = 30

# Common Crawl's list of crawls (newest first) and the address patterns to look up.
COMMON_CRAWL_INFO_URL = "https://index.commoncrawl.org/collinfo.json"
BOARD_ADDRESSES: dict[str, Source] = {
    "boards.greenhouse.io": "greenhouse",
    "job-boards.greenhouse.io": "greenhouse",
    "jobs.lever.co": "lever",
    "jobs.ashbyhq.com": "ashby",
    "apply.workable.com": "workable",
}

# Some boards put the company in the SUBDOMAIN (the part before the main site
# name) instead of the path: "bunq.recruitee.com", not "recruitee.com/bunq".
SUBDOMAIN_BOARDS: dict[str, Source] = {
    "recruitee.com": "recruitee",
    "jobs.personio.de": "personio",
    "jobs.personio.com": "personio",
}

# Subdomains of those sites that are the board's own pages, not companies.
NOT_COMPANY_SUBDOMAINS = {"www", "api", "app", "blog", "help", "support", "status"}

# The API address that answers for one company, per board. Same as the fetchers.
CHECK_URLS: dict[str, str] = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
    "lever": "https://api.lever.co/v0/postings/{slug}?mode=json&limit=1",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{slug}",
    "workable": "https://apply.workable.com/api/v1/widget/accounts/{slug}",
    "recruitee": "https://{slug}.recruitee.com/api/offers/",
    "personio": "https://{slug}.jobs.personio.de/xml",
}

# First parts of board addresses that are pages, not companies.
NOT_COMPANIES = {
    "",
    "embed",
    "v1",
    "api",
    "static",
    "assets",
    "favicon.ico",
    "robots.txt",
}


class Company(BaseModel):
    """One job board: which platform, and the company's slug on it."""

    # frozen=True makes Company values unchangeable, which lets Python put
    # them in a set (sets need values that can't change).
    model_config = {"frozen": True}

    source: Source
    slug: str


class DiscoveryReport(BaseModel):
    """What one discovery run did."""

    candidates: int  # slugs found, before checking
    already_known: int  # already in companies.csv, so not checked again
    added: list[Company]
    not_found: int  # the API said 404: not a real board
    errors: int  # couldn't check (network trouble); will be tried next time


# ---------------------------------------------------------------------------
# The saved list: companies.csv
# ---------------------------------------------------------------------------


def companies_path() -> Path:
    """Where the list lives. Override with JOBS_SEARCH_COMPANIES_FILE."""
    return Path(os.environ.get("JOBS_SEARCH_COMPANIES_FILE", "companies.csv"))


def load_companies(path: Path) -> list[Company]:
    """Read companies.csv. A missing file just means an empty list."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [Company(source=row["source"], slug=row["slug"]) for row in rows]


def save_companies(path: Path, companies: list[Company]) -> None:
    """Write companies.csv, sorted and without duplicates (easy to read in git)."""
    unique = sorted(set(companies), key=lambda company: (company.source, company.slug))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "slug"])
        writer.writeheader()
        for company in unique:
            writer.writerow({"source": company.source, "slug": company.slug})


# ---------------------------------------------------------------------------
# Finding candidates
# ---------------------------------------------------------------------------


def looks_like_slug(text: str) -> bool:
    """Slugs are short and use only letters, digits, - and _ (e.g. "gitlab")."""
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    return 1 < len(text) <= 60 and set(text) <= allowed


def company_from_url(url: str) -> Company | None:
    """Turn a job-board address into a Company, or None if it isn't one.

    >>> company_from_url("https://jobs.lever.co/spotify/1a2b-3c4d")
    Company(source='lever', slug='spotify')
    >>> company_from_url("https://boards.greenhouse.io/embed/job_board?for=gitlab")
    Company(source='greenhouse', slug='gitlab')
    >>> company_from_url("https://bunq.recruitee.com/o/security-engineer")
    Company(source='recruitee', slug='bunq')
    """
    parts = urlparse(url)  # splits a URL into host, path, query, ...
    host = parts.netloc.lower().removeprefix("www.")

    # Subdomain boards: the slug is the start of the host, "bunq.recruitee.com" -> "bunq".
    for domain, subdomain_source in SUBDOMAIN_BOARDS.items():
        if host.endswith("." + domain):
            slug = host.removesuffix("." + domain)
            # "a.b.recruitee.com" would leave "a.b": not a slug, so it's dropped.
            if slug in NOT_COMPANY_SUBDOMAINS or not looks_like_slug(slug):
                return None
            return Company(source=subdomain_source, slug=slug)

    # Path boards: the host must be one we know.
    source = BOARD_ADDRESSES.get(host)
    if source is None:
        return None

    # The slug is the first piece of the path: /spotify/1a2b -> "spotify".
    slug = parts.path.strip("/").split("/")[0].lower()
    # Greenhouse "embed" pages name the company in the query instead: ?for=gitlab
    if slug == "embed":
        slug = parse_qs(parts.query).get("for", [""])[0].lower()

    if slug in NOT_COMPANIES or not looks_like_slug(slug):
        return None
    return Company(source=source, slug=slug)


def parse_index_lines(text: str) -> set[Company]:
    """Read Common Crawl's answer: one JSON object per line, each with a "url"."""
    found = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue  # blank lines or an error page
        url = json.loads(line).get("url", "")
        company = company_from_url(url)
        if company is not None:
            found.add(company)
    return found


def common_crawl_candidates(pages: int = 1) -> set[Company]:
    """Ask Common Crawl's newest index for job-board addresses.

    Each "page" of results is a few thousand addresses; more pages find more
    companies but take longer. Common Crawl is free and often busy, so if it
    doesn't answer this returns what it got (maybe nothing) instead of
    failing: the seed list and next run still work.
    """
    found = set()
    try:
        crawls = httpx.get(COMMON_CRAWL_INFO_URL, timeout=TIMEOUT_SECONDS).json()
        index_url = crawls[0]["cdx-api"]  # [0] is the newest crawl
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
        print(f"  Common Crawl isn't answering ({type(error).__name__}); skipping it.")
        return found

    # The address patterns to ask for. "jobs.lever.co/*" means every page under
    # jobs.lever.co, and "*.recruitee.com" means every subdomain of recruitee.com.
    patterns = []
    for address in BOARD_ADDRESSES:
        patterns.append(f"{address}/*")
    for domain in SUBDOMAIN_BOARDS:
        patterns.append(f"*.{domain}")

    for pattern in patterns:
        for page in range(pages):
            params = {
                "url": pattern,
                "output": "json",
                "fl": "url",
                "page": page,
            }
            try:
                response = httpx.get(index_url, params=params, timeout=120)
                response.raise_for_status()
            except httpx.HTTPError as error:
                print(
                    f"  Common Crawl: no answer for {pattern} ({type(error).__name__})."
                )
                break  # move on to the next pattern
            found |= parse_index_lines(response.text)  # |= adds to the set
    return found


def seed_candidates(path: Path) -> set[Company]:
    """Read company_seeds.txt: lines like "lever spotify". # starts a comment."""
    found = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()  # drop comments and spaces
        if not line:
            continue
        words = line.split()
        if len(words) != 2:
            print(f"  Skipping seed line {line!r}: expected '<board> <slug>'.")
            continue
        source, slug = words
        if source in CHECK_URLS and looks_like_slug(slug.lower()):
            found.add(Company(source=source, slug=slug.lower()))
    return found


# ---------------------------------------------------------------------------
# Checking candidates
# ---------------------------------------------------------------------------


def board_exists(company: Company) -> bool:
    """Ask the real API. True for a 200 answer, False for 404.

    Raises:
        httpx.HTTPError: for anything else (a network problem, a 500 error),
            so the caller can try this company again next time.
    """
    url = CHECK_URLS[company.source].format(slug=company.slug)
    response = httpx.get(url, timeout=TIMEOUT_SECONDS)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def discover(
    path: Path,
    candidates: set[Company],
    check: Callable[[Company], bool] | None = None,
    pause_seconds: float = 0.2,
) -> DiscoveryReport:
    """Check every new candidate, and add the real boards to companies.csv.

    `check` is passed in so tests can use a fake instead of the network.
    `pause_seconds` waits between checks, to be polite to the APIs.
    """
    # Why not `check=board_exists` above? A default value is fixed once, when
    # Python first reads the function, so a test that swaps board_exists
    # later wouldn't change it. Looking it up here, at call time, does.
    if check is None:
        check = board_exists
    known = set(load_companies(path))
    to_check = sorted(
        candidates - known,  # set difference: candidates we don't already have
        key=lambda company: (company.source, company.slug),
    )

    added = []
    not_found = 0
    errors = 0
    for number, company in enumerate(to_check, start=1):
        print(f"  [{number}/{len(to_check)}] {company.source} {company.slug}")
        try:
            if check(company):
                added.append(company)
            else:
                not_found += 1
        except httpx.HTTPError:
            errors += 1
        time.sleep(pause_seconds)

    save_companies(path, list(known) + added)
    return DiscoveryReport(
        candidates=len(candidates),
        already_known=len(candidates & known),  # set intersection: in both
        added=added,
        not_found=not_found,
        errors=errors,
    )


def remove_companies(path: Path, gone: list[Company]) -> None:
    """Take boards off the list, e.g. ones that started answering 404."""
    remaining = [company for company in load_companies(path) if company not in gone]
    save_companies(path, remaining)

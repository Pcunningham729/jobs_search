"""Fetch job postings from Greenhouse and turn them into `Job` objects.

Greenhouse is a hiring system many companies use for their careers pages. It
has a public API (a web address that returns data instead of a web page) that
needs no login:

    https://boards-api.greenhouse.io/v1/boards/<company>/jobs

The work is split into two steps on purpose:

- `parse_greenhouse()` translates JSON we already have into `Job` objects.
  It never touches the network, so tests can feed it a saved response.
- `fetch_greenhouse()` downloads the JSON, then hands it to `parse_greenhouse()`.
"""

# `html` is built into Python. We use it to undo HTML "escaping" (see below).
import html

# httpx is a library for making web requests, the way a browser does.
import httpx

from jobs_search.fetchers.helpers import html_to_text
from jobs_search.models import Job

# The API address. `{company}` is a placeholder that `.format()` fills in later.
API_URL = "https://boards-api.greenhouse.io/v1/boards/{company}/jobs"

# How many seconds to wait for Greenhouse to answer before giving up.
# Without a timeout, a stuck request could make the program hang forever.
TIMEOUT_SECONDS = 30


def description_to_text(escaped_html: str) -> str:
    """Turn Greenhouse's job description into plain text.

    Greenhouse "escapes" its HTML: instead of `<p>` it sends `&lt;p&gt;`
    (`&lt;` is HTML's code for `<`, and `&gt;` for `>`). So this takes two steps:

        "&lt;p&gt;Build &lt;b&gt;pipelines&lt;/b&gt;&lt;/p&gt;"
        --unescape-->   "<p>Build <b>pipelines</b></p>"
        --strip tags--> "Build pipelines"
    """
    # Step 1: turn `&lt;p&gt;` back into real HTML, `<p>`.
    real_html = html.unescape(escaped_html)

    # Step 2: strip the tags, using the helper every fetcher shares.
    return html_to_text(real_html)


def parse_salary(job_data: dict) -> tuple[int | None, int | None]:
    """Find a job's yearly salary range in US dollars.

    Returns two values: (lowest salary, highest salary). Both are None if the
    posting has no salary in US dollars.

    Greenhouse only includes `pay_input_ranges` when we ask for it (see
    `fetch_greenhouse`), and gives amounts in cents: 13920000 cents is $139,200.
    """
    # `.get(...)` returns None instead of crashing when a field is missing.
    # `or []` then swaps that None for an empty list, so the loop below simply
    # runs zero times.
    pay_ranges = job_data.get("pay_input_ranges") or []

    # A job can list several ranges (for example one per country). Use the
    # first one in US dollars.
    for pay_range in pay_ranges:
        if pay_range.get("currency_type") == "USD":
            # `//` divides and drops anything after the decimal point.
            lowest = pay_range["min_cents"] // 100
            highest = pay_range["max_cents"] // 100
            return lowest, highest

    # No US-dollar range was found.
    return None, None


def parse_greenhouse(data: dict, company: str) -> list[Job]:
    """Turn a Greenhouse API response into a list of `Job` objects.

    `data` is the response JSON, already turned into Python dicts and lists.
    It looks like: {"jobs": [{...one job...}, {...another job...}], "meta": {...}}
    `company` is the board name we asked for, for example "gitlab".
    """
    jobs = []
    for job_data in data["jobs"]:
        salary_min, salary_max = parse_salary(job_data)

        job = Job(
            id=f"greenhouse:{company}:{job_data['id']}",
            source="greenhouse",
            company=company,
            title=job_data["title"],
            # `location` is itself a small dict, like {"name": "Remote, United States"}.
            location=job_data["location"]["name"],
            url=job_data["absolute_url"],
            description=description_to_text(job_data["content"]),
            salary_min=salary_min,
            salary_max=salary_max,
            # Dates are text like "2026-11-06T15:25:07-05:00".
            # Pydantic turns them into datetime objects.
            posted_at=job_data["first_published"],
            updated_at=job_data["updated_at"],
        )
        jobs.append(job)
    return jobs


def fetch_greenhouse(company: str) -> list[Job]:
    """Download all open jobs for one company from Greenhouse.

    `company` is the company's Greenhouse board name: the part after
    `greenhouse.io/` in its job links, for example "gitlab".
    """
    url = API_URL.format(company=company)

    # `params` become the `?content=true&pay_transparency=true` part of the
    # address. content=true asks for full descriptions, and
    # pay_transparency=true asks for salary ranges.
    response = httpx.get(
        url,
        params={"content": "true", "pay_transparency": "true"},
        timeout=TIMEOUT_SECONDS,
    )

    # If Greenhouse answered with an error (for example 404 Not Found, which
    # happens when the company name is wrong), stop here with a clear error
    # instead of trying to read a response that has no jobs in it.
    response.raise_for_status()

    # `.json()` turns the JSON text into Python dicts and lists.
    return parse_greenhouse(response.json(), company)

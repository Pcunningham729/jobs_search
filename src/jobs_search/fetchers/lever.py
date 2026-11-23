"""Fetch job postings from Lever and turn them into `Job` objects.

Lever is another hiring system companies use for their careers pages. Its
public API needs no login:

    https://api.lever.co/v0/postings/<company>?mode=json

Same two-step pattern as greenhouse.py:
- `parse_lever()` translates JSON we already have into `Job` objects (no network).
- `fetch_lever()` downloads the JSON, then hands it to `parse_lever()`.

How Lever differs from Greenhouse:
- The response is a plain list of jobs, not {"jobs": [...]}.
- A description comes in three parts: an intro, some titled lists
  ("What You'll Do", "Who You Are"...), and a closing section. We join them.
- Remote is a separate field (`workplaceType`), not part of the location text.
- The posted date is a number: milliseconds since 1 January 1970 (see below).
"""

from datetime import UTC, datetime

import httpx

from jobs_search.fetchers.helpers import build_location, html_to_text
from jobs_search.models import Job

API_URL = "https://api.lever.co/v0/postings/{company}"
TIMEOUT_SECONDS = 30


def build_description(job_data: dict) -> str:
    """Join Lever's three description parts into one plain-text description.

    - `descriptionPlain`: the intro, already plain text.
    - `lists`: titled sections like "What You'll Do". Their content is HTML,
      so we convert it with html_to_text().
    - `additionalPlain`: a closing section (often the salary and benefits),
      already plain text.
    """
    parts = [job_data.get("descriptionPlain", "")]

    for section in job_data.get("lists", []):
        parts.append(section["text"])  # the section's title
        parts.append(html_to_text(section["content"]))

    parts.append(job_data.get("additionalPlain", ""))

    # Keep only the parts that have text, and put a line break between them.
    non_empty_parts = []
    for part in parts:
        part = part.strip()
        if part:
            non_empty_parts.append(part)
    return "\n".join(non_empty_parts)


def parse_salary(job_data: dict) -> tuple[int | None, int | None]:
    """Find a job's yearly salary range in US dollars: (lowest, highest).

    Lever only sometimes fills in `salaryRange`, which looks like:
        {"min": 75645, "max": 94556, "currency": "USD", "interval": "per-year-salary"}
    Amounts are already whole dollars (not cents like Greenhouse).
    Returns (None, None) if there's no yearly salary in US dollars, for example
    when the range is an hourly wage.
    """
    salary = job_data.get("salaryRange")
    if salary is None:
        return None, None

    if salary.get("currency") == "USD" and salary.get("interval") == "per-year-salary":
        return salary["min"], salary["max"]

    return None, None


def milliseconds_to_datetime(milliseconds: int) -> datetime:
    """Turn Lever's date number into a datetime.

    Lever counts milliseconds since midnight on 1 January 1970, in UTC
    (a common way computers store time). For example, 1794565648197 is
    2026-11-13 at 10:27:28 UTC. Python's `fromtimestamp` expects *seconds*,
    so we divide by 1000 first.
    """
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def parse_lever(data: list[dict], company: str) -> list[Job]:
    """Turn a Lever API response into a list of `Job` objects.

    `data` is the response JSON: a list with one dict per job.
    `company` is the company's Lever name, for example "spotify".
    """
    jobs = []
    for job_data in data:
        salary_min, salary_max = parse_salary(job_data)

        # `categories` holds the location. `allLocations` lists every place
        # the job can be done. Older postings may only have `location`.
        categories = job_data["categories"]
        places = categories.get("allLocations") or [categories.get("location", "")]
        is_remote = job_data.get("workplaceType") == "remote"

        job = Job(
            id=f"lever:{company}:{job_data['id']}",
            source="lever",
            company=company,
            title=job_data["text"].strip(),  # Lever calls the title "text"
            location=build_location(places, is_remote),
            url=job_data["hostedUrl"],
            description=build_description(job_data),
            salary_min=salary_min,
            salary_max=salary_max,
            posted_at=milliseconds_to_datetime(job_data["createdAt"]),
            # Lever doesn't say when a posting was last changed.
            updated_at=None,
        )
        jobs.append(job)
    return jobs


def fetch_lever(company: str) -> list[Job]:
    """Download all open jobs for one company from Lever.

    `company` is the company's Lever name: the part after `jobs.lever.co/`
    in its job links, for example "spotify".
    """
    url = API_URL.format(company=company)

    # mode=json asks Lever for data instead of a web page.
    response = httpx.get(url, params={"mode": "json"}, timeout=TIMEOUT_SECONDS)

    # Stop with a clear error if Lever answered with one (e.g. 404 for a wrong name).
    response.raise_for_status()

    return parse_lever(response.json(), company)

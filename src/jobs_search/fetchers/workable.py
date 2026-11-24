"""Fetch job postings from Workable and turn them into `Job` objects.

Workable is a hiring system used by many small and mid-size companies. Its
careers pages (apply.workable.com/<company>) are powered by a public "widget"
API that needs no login:

    https://apply.workable.com/api/v1/widget/accounts/<company>?details=true

Same two-step pattern as greenhouse.py:
- `parse_workable()` translates JSON we already have into `Job` objects (no network).
- `fetch_workable()` downloads the JSON, then hands it to `parse_workable()`.

How Workable differs from the others:
- Without `details=true` the answer has no descriptions, so we always ask for it.
- Each location is split into pieces (city, region, country) that we join.
- Remote is a separate true/false field (`telecommuting`).
- `published_on` is only a date ("2026-05-29"), with no time of day.
- No salary information.
"""

import httpx

from jobs_search.fetchers.helpers import build_location, html_to_text
from jobs_search.models import Job

API_URL = "https://apply.workable.com/api/v1/widget/accounts/{company}"
TIMEOUT_SECONDS = 30


def location_to_text(location: dict) -> str:
    """Join one Workable location's pieces into text, skipping empty ones.

    Example:
        {"city": "New York", "region": "New York", "country": "United States"}
            -> "New York, New York, United States"
        {"city": "", "region": None, "country": "France"}
            -> "France"
    """
    pieces = []
    for key in ["city", "region", "country"]:
        # `or ""` turns a missing value (None) into an empty string.
        piece = (location.get(key) or "").strip()
        if piece:
            pieces.append(piece)
    return ", ".join(pieces)


def parse_workable(data: dict, company: str) -> list[Job]:
    """Turn a Workable API response into a list of `Job` objects.

    `data` is the response JSON: {"name": ..., "description": ..., "jobs": [...]}
    `company` is the company's Workable name, for example "huggingface".
    """
    jobs = []
    for job_data in data["jobs"]:
        # A job can list several locations. A "hidden" location means the
        # company doesn't show the exact place, so we keep only its country.
        places = []
        for location in job_data.get("locations", []):
            if location.get("hidden", False):
                places.append((location.get("country") or "").strip())
            else:
                places.append(location_to_text(location))
        is_remote = job_data.get("telecommuting", False)

        job = Job(
            # `shortcode` is Workable's id for a job, like "19A136F8E2".
            id=f"workable:{company}:{job_data['shortcode']}",
            source="workable",
            company=company,
            title=job_data["title"].strip(),
            location=build_location(places, is_remote),
            url=job_data["url"],
            description=html_to_text(job_data.get("description", "")),
            # Workable doesn't share salaries or when a posting last changed.
            salary_min=None,
            salary_max=None,
            # A date like "2026-05-29". Pydantic turns it into a datetime at midnight.
            posted_at=job_data.get("published_on"),
            updated_at=None,
        )
        jobs.append(job)
    return jobs


def fetch_workable(company: str) -> list[Job]:
    """Download all open jobs for one company from Workable.

    `company` is the company's Workable name: the part after
    `apply.workable.com/` in its job links, for example "huggingface".
    """
    url = API_URL.format(company=company)

    # details=true asks Workable to include each job's description.
    response = httpx.get(url, params={"details": "true"}, timeout=TIMEOUT_SECONDS)

    # Stop with a clear error if Workable answered with one (e.g. 404 for a wrong name).
    response.raise_for_status()

    return parse_workable(response.json(), company)

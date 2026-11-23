"""Fetch job postings from Ashby and turn them into `Job` objects.

Ashby is another hiring system companies use for their careers pages. Its
public API needs no login:

    https://api.ashbyhq.com/posting-api/job-board/<company>?includeCompensation=true

Same two-step pattern as greenhouse.py:
- `parse_ashby()` translates JSON we already have into `Job` objects (no network).
- `fetch_ashby()` downloads the JSON, then hands it to `parse_ashby()`.

How Ashby differs from Greenhouse:
- Descriptions already come as plain text (`descriptionPlain`), so no HTML work.
- A job has a main `location` plus a list of `secondaryLocations`.
- Remote is a separate field (`workplaceType`), not part of the location text.
- Salary is in whole dollars, inside a `compensation` section, and only
  included when we ask with includeCompensation=true.
"""

import httpx

from jobs_search.fetchers.helpers import build_location
from jobs_search.models import Job

API_URL = "https://api.ashbyhq.com/posting-api/job-board/{company}"
TIMEOUT_SECONDS = 30


def parse_salary(job_data: dict) -> tuple[int | None, int | None]:
    """Find a job's yearly salary range in US dollars: (lowest, highest).

    Ashby lists pay as "components" (salary, equity, bonus...), each like:
        {"compensationType": "Salary", "currencyCode": "USD",
         "interval": "1 YEAR", "minValue": 211400, "maxValue": 290600}
    We look for the yearly US-dollar salary and ignore the rest.
    Returns (None, None) if there isn't one.
    """
    compensation = job_data.get("compensation") or {}
    components = compensation.get("summaryComponents") or []

    for component in components:
        is_salary = component.get("compensationType") == "Salary"
        is_usd = component.get("currencyCode") == "USD"
        is_yearly = component.get("interval") == "1 YEAR"

        if is_salary and is_usd and is_yearly:
            lowest = component.get("minValue")
            highest = component.get("maxValue")
            # Ashby may send numbers like 211400.0. `int()` makes them whole
            # numbers, but only if a value is actually there.
            if lowest is not None:
                lowest = int(lowest)
            if highest is not None:
                highest = int(highest)
            return lowest, highest

    return None, None


def parse_ashby(data: dict, company: str) -> list[Job]:
    """Turn an Ashby API response into a list of `Job` objects.

    `data` is the response JSON: {"jobs": [{...one job...}, ...], "apiVersion": ...}
    `company` is the company's Ashby name, for example "ramp".
    """
    jobs = []
    for job_data in data["jobs"]:
        # Companies can hide a posting from their board. Skip those.
        if not job_data.get("isListed", True):
            continue  # `continue` jumps straight to the next job in the loop

        salary_min, salary_max = parse_salary(job_data)

        # The main location, then any extra ones.
        places = [job_data.get("location", "")]
        for extra in job_data.get("secondaryLocations", []):
            places.append(extra.get("location", ""))
        is_remote = job_data.get("workplaceType") == "Remote"

        job = Job(
            id=f"ashby:{company}:{job_data['id']}",
            source="ashby",
            company=company,
            # `.strip()` because some Ashby titles start with a stray space.
            title=job_data["title"].strip(),
            location=build_location(places, is_remote),
            url=job_data["jobUrl"],
            description=job_data["descriptionPlain"].strip(),
            salary_min=salary_min,
            salary_max=salary_max,
            # Text like "2026-04-07T17:12:35.753+00:00". Pydantic turns it into a datetime.
            posted_at=job_data.get("publishedAt"),
            # Ashby doesn't say when a posting was last changed.
            updated_at=None,
        )
        jobs.append(job)
    return jobs


def fetch_ashby(company: str) -> list[Job]:
    """Download all open jobs for one company from Ashby.

    `company` is the company's Ashby name: the part after `jobs.ashbyhq.com/`
    in its job links, for example "ramp".
    """
    url = API_URL.format(company=company)

    # includeCompensation=true asks Ashby to include salary ranges.
    response = httpx.get(
        url, params={"includeCompensation": "true"}, timeout=TIMEOUT_SECONDS
    )

    # Stop with a clear error if Ashby answered with one (e.g. 404 for a wrong name).
    response.raise_for_status()

    return parse_ashby(response.json(), company)

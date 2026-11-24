"""Fetch job postings from Recruitee and turn them into `Job` objects.

Recruitee is a hiring system popular with European companies. Each company
gets its own SUBDOMAIN (the part before the main site name), and its public
API needs no login:

    https://<company>.recruitee.com/api/offers/

Same two-step pattern as greenhouse.py:
- `parse_recruitee()` translates JSON we already have into `Job` objects (no network).
- `fetch_recruitee()` downloads the JSON, then hands it to `parse_recruitee()`.

How Recruitee differs from the others:
- Jobs are called "offers".
- The description comes in two HTML parts: `description` and `requirements`.
- Dates look like "2026-10-22 06:17:49 UTC", which Pydantic can't read as-is.
- Salary amounts are text ("100000"), with a currency and a period
  ("year", "month"...). We only keep yearly US-dollar salaries.
"""

import httpx

from jobs_search.fetchers.helpers import build_location, html_to_text
from jobs_search.models import Job

API_URL = "https://{company}.recruitee.com/api/offers/"
TIMEOUT_SECONDS = 30


def parse_date(text: str | None) -> str | None:
    """Rewrite Recruitee's date text into a form Pydantic understands.

    Example:
        "2026-10-22 06:17:49 UTC"  ->  "2026-10-22 06:17:49+00:00"

    "+00:00" is the standard way to write "zero hours from UTC".
    Returns None if there's no date.
    """
    if not text:
        return None
    return text.replace(" UTC", "+00:00")


def parse_salary(offer: dict) -> tuple[int | None, int | None]:
    """Find an offer's yearly salary range in US dollars: (lowest, highest).

    Recruitee sends salary like:
        {"min": "100000", "max": "120000", "period": "year", "currency": "EUR"}
    Returns (None, None) unless it's a yearly salary in US dollars.
    """
    salary = offer.get("salary") or {}
    if salary.get("currency") != "USD" or salary.get("period") != "year":
        return None, None

    lowest = salary.get("min")
    highest = salary.get("max")
    # The amounts are text, so `int()` turns them into whole numbers, but only
    # if a value is actually there.
    if lowest is not None:
        lowest = int(float(lowest))
    if highest is not None:
        highest = int(float(highest))
    return lowest, highest


def parse_recruitee(data: dict, company: str) -> list[Job]:
    """Turn a Recruitee API response into a list of `Job` objects.

    `data` is the response JSON: {"offers": [{...one job...}, ...]}
    `company` is the company's Recruitee name, for example "bunq".
    """
    jobs = []
    for offer in data["offers"]:
        # Only published offers are real, open jobs.
        if offer.get("status", "published") != "published":
            continue

        salary_min, salary_max = parse_salary(offer)

        # For remote jobs, Recruitee's `location` is just "Remote job", which
        # doesn't say which country. Adding the country keeps that information
        # for the location filter: "Remote job; France".
        is_remote = offer.get("remote", False)
        places = [offer.get("location") or ""]
        if is_remote:
            places.append(offer.get("country") or "")

        # The description is in two HTML parts. Join them, then strip the tags.
        description_html = (offer.get("description") or "") + (
            offer.get("requirements") or ""
        )

        job = Job(
            id=f"recruitee:{company}:{offer['id']}",
            source="recruitee",
            company=company,
            title=offer["title"].strip(),
            location=build_location(places, is_remote),
            url=offer["careers_url"],
            description=html_to_text(description_html),
            salary_min=salary_min,
            salary_max=salary_max,
            posted_at=parse_date(offer.get("published_at")),
            updated_at=parse_date(offer.get("updated_at")),
        )
        jobs.append(job)
    return jobs


def fetch_recruitee(company: str) -> list[Job]:
    """Download all open jobs for one company from Recruitee.

    `company` is the company's Recruitee name: the part before
    `.recruitee.com` in its careers address, for example "bunq".
    """
    url = API_URL.format(company=company)
    response = httpx.get(url, timeout=TIMEOUT_SECONDS)

    # Stop with a clear error if Recruitee answered with one (e.g. 404 for a wrong name).
    response.raise_for_status()

    return parse_recruitee(response.json(), company)

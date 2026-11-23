"""Tests for fetchers/ashby.py.

Offline, like the Greenhouse tests: they read 3 real Ramp jobs saved in
tests/fixtures/ashby.json, and fake the download with `monkeypatch`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from jobs_search.fetchers import FETCHERS, ashby
from jobs_search.fetchers.ashby import fetch_ashby, parse_ashby, parse_salary

FIXTURE = Path("tests/fixtures/ashby.json")


def load_fixture() -> dict:
    """Read the saved Ashby response and turn the JSON into Python data."""
    with open(FIXTURE, encoding="utf-8") as file:
        return json.load(file)


def test_parse_ashby_reads_every_job():
    assert len(parse_ashby(load_fixture(), company="ramp")) == 3


def test_parse_ashby_maps_fields():
    first = parse_ashby(load_fixture(), company="ramp")[0]

    assert first.id == "ashby:ramp:d84bbf19-572a-499c-9c87-0c154ce85caf"
    assert first.source == "ashby"
    assert first.company == "ramp"
    assert first.title == "Account Manager | Mid-Market"
    # Main location first, then the secondary locations.
    assert first.location == "New York, NY (HQ); San Francisco, CA; Remote (US)"
    assert first.url == (
        "https://jobs.ashbyhq.com/ramp/d84bbf19-572a-499c-9c87-0c154ce85caf"
    )
    assert first.salary_min == 234000
    assert first.salary_max == 260000
    # "publishedAt": "2026-09-23T16:15:40.412+00:00" (+00:00 means UTC).
    assert first.posted_at == datetime(2026, 9, 23, 16, 15, 40, 412000, tzinfo=UTC)
    assert first.updated_at is None


def test_title_has_no_stray_spaces():
    # In the fixture this title is " Software Engineer, Security, Cloud".
    second = parse_ashby(load_fixture(), company="ramp")[1]

    assert second.title == "Software Engineer, Security, Cloud"


def test_job_without_salary_has_no_salary():
    third = parse_ashby(load_fixture(), company="ramp")[2]

    assert third.salary_min is None
    assert third.salary_max is None


def test_unlisted_jobs_are_skipped():
    data = load_fixture()
    data["jobs"][0]["isListed"] = False  # pretend the company hid this one

    jobs = parse_ashby(data, company="ramp")

    assert len(jobs) == 2
    assert jobs[0].title == "Software Engineer, Security, Cloud"


def test_parse_salary_skips_equity_and_other_currencies():
    job_data = {
        "compensation": {
            "summaryComponents": [
                # Equity, not salary: skipped.
                {"compensationType": "EquityPercentage", "currencyCode": None},
                # Salary, but in euros: skipped.
                {
                    "compensationType": "Salary",
                    "currencyCode": "EUR",
                    "interval": "1 YEAR",
                    "minValue": 50000,
                    "maxValue": 70000,
                },
            ]
        }
    }

    assert parse_salary(job_data) == (None, None)


class FakeResponse:
    """Pretends to be an httpx response, without any network."""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self.data


def test_fetch_ashby_requests_the_right_address(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(load_fixture())

    monkeypatch.setattr(ashby.httpx, "get", fake_get)

    jobs = fetch_ashby("ramp")

    assert len(jobs) == 3
    assert calls[0]["url"] == "https://api.ashbyhq.com/posting-api/job-board/ramp"
    assert calls[0]["params"] == {"includeCompensation": "true"}
    assert calls[0]["timeout"] == 30


def test_ashby_is_registered():
    assert FETCHERS["ashby"] is fetch_ashby

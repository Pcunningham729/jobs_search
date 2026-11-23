"""Tests for fetchers/lever.py.

Offline, like the Greenhouse tests: they read 3 real Spotify jobs saved in
tests/fixtures/lever.json, and fake the download with `monkeypatch`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from jobs_search.fetchers import FETCHERS, lever
from jobs_search.fetchers.lever import fetch_lever, parse_lever, parse_salary

FIXTURE = Path("tests/fixtures/lever.json")


def load_fixture() -> list[dict]:
    """Read the saved Lever response. Lever returns a list, not a dict."""
    with open(FIXTURE, encoding="utf-8") as file:
        return json.load(file)


def test_parse_lever_reads_every_job():
    assert len(parse_lever(load_fixture(), company="spotify")) == 3


def test_parse_lever_maps_fields():
    first = parse_lever(load_fixture(), company="spotify")[0]

    assert first.id == "lever:spotify:b24fe5e5-dcb2-4238-a883-5fdb6bd404db"
    assert first.source == "lever"
    assert first.company == "spotify"
    assert first.title == "Accounting Analyst, Content Accounting"
    assert first.location == "New York, NY"
    assert first.url == (
        "https://jobs.lever.co/spotify/b24fe5e5-dcb2-4238-a883-5fdb6bd404db"
    )
    # This is the one job in the fixture with a salaryRange.
    assert first.salary_min == 75645
    assert first.salary_max == 94556
    # createdAt 1794565648197 milliseconds = 13 Nov 2026, 10:27:28.197 UTC.
    assert first.posted_at == datetime(2026, 11, 13, 10, 27, 28, 197000, tzinfo=UTC)
    assert first.updated_at is None


def test_remote_job_says_remote_in_location():
    # The second job has workplaceType "remote" but its places are cities.
    second = parse_lever(load_fixture(), company="spotify")[1]

    assert second.location == "Remote; New York, NY; San Francisco, CA"


def test_description_includes_the_lists_as_plain_text():
    second = parse_lever(load_fixture(), company="spotify")[1]

    # Text from the intro...
    assert second.description.startswith("Spotify Technology is our business")
    # ...and a list title, with the list's HTML turned into text.
    assert "What You'll Do" in second.description
    assert "<" not in second.description


def test_job_without_salary_range_has_no_salary():
    third = parse_lever(load_fixture(), company="spotify")[2]

    assert third.salary_min is None
    assert third.salary_max is None


def test_parse_salary_ignores_hourly_wages():
    job_data = {
        "salaryRange": {
            "min": 25,
            "max": 30,
            "currency": "USD",
            "interval": "per-hour-wage",
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


def test_fetch_lever_requests_the_right_address(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(load_fixture())

    monkeypatch.setattr(lever.httpx, "get", fake_get)

    jobs = fetch_lever("spotify")

    assert len(jobs) == 3
    assert calls[0]["url"] == "https://api.lever.co/v0/postings/spotify"
    assert calls[0]["params"] == {"mode": "json"}
    assert calls[0]["timeout"] == 30


def test_lever_is_registered():
    assert FETCHERS["lever"] is fetch_lever

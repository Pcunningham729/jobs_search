"""Tests for fetchers/workable.py.

Offline, like the Greenhouse tests: they read 3 real Hugging Face jobs saved
in tests/fixtures/workable.json, and fake the download with `monkeypatch`.
"""

import json
from datetime import datetime
from pathlib import Path

from jobs_search.fetchers import FETCHERS, workable
from jobs_search.fetchers.workable import (
    fetch_workable,
    location_to_text,
    parse_workable,
)

FIXTURE = Path("tests/fixtures/workable.json")


def load_fixture() -> dict:
    """Read the saved Workable response and turn the JSON into Python data."""
    with open(FIXTURE, encoding="utf-8") as file:
        return json.load(file)


def test_parse_workable_reads_every_job():
    assert len(parse_workable(load_fixture(), company="huggingface")) == 3


def test_parse_workable_maps_fields():
    first = parse_workable(load_fixture(), company="huggingface")[0]

    assert first.id == "workable:huggingface:19A136F8E2"
    assert first.source == "workable"
    assert first.company == "huggingface"
    assert first.title == "Open-Source Machine Learning Engineer - US Remote"
    # "telecommuting": true, so "Remote" is added in front of the place.
    assert first.location == "Remote; New York, New York, United States"
    assert first.url == "https://apply.workable.com/j/19A136F8E2"
    assert first.salary_min is None
    assert first.salary_max is None
    # "published_on": "2026-05-29" has no time, so it becomes midnight.
    assert first.posted_at == datetime(2026, 5, 29)
    assert first.updated_at is None


def test_hidden_location_keeps_only_the_country():
    # The second job's only location is hidden, with country "United States".
    second = parse_workable(load_fixture(), company="huggingface")[1]

    assert second.location == "Remote; United States"


def test_description_is_plain_text():
    first = parse_workable(load_fixture(), company="huggingface")[0]

    assert "<" not in first.description
    assert "At Hugging Face, we're on a journey to democratize good AI" in (
        first.description
    )


def test_location_to_text_skips_empty_pieces():
    assert location_to_text({"city": "", "region": None, "country": "France"}) == (
        "France"
    )
    assert location_to_text({"city": "Paris", "country": "France"}) == "Paris, France"


def test_office_job_is_not_marked_remote():
    # A made-up job that isn't remote.
    data = {
        "jobs": [
            {
                "shortcode": "ABC123",
                "title": "Engineer",
                "url": "https://apply.workable.com/j/ABC123",
                "telecommuting": False,
                "locations": [{"city": "Berlin", "country": "Germany"}],
            }
        ]
    }

    job = parse_workable(data, company="acme")[0]

    assert job.location == "Berlin, Germany"
    assert job.description == ""  # no description in the data
    assert job.posted_at is None  # no date in the data


class FakeResponse:
    """Pretends to be the response httpx.get() returns, without any network."""

    def __init__(self, data: dict):
        self.data = data

    def raise_for_status(self) -> None:
        pass  # pretend the request worked

    def json(self) -> dict:
        return self.data


def test_fetch_workable_requests_the_right_address(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(load_fixture())

    monkeypatch.setattr(workable.httpx, "get", fake_get)

    jobs = fetch_workable("huggingface")

    assert len(jobs) == 3
    assert calls[0]["url"] == (
        "https://apply.workable.com/api/v1/widget/accounts/huggingface"
    )
    assert calls[0]["params"] == {"details": "true"}
    assert calls[0]["timeout"] == 30


def test_workable_is_registered():
    assert FETCHERS["workable"] is fetch_workable

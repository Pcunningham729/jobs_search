"""Tests for fetchers/greenhouse.py.

These tests never use the network. They read a real Greenhouse response that
was saved earlier (tests/fixtures/greenhouse.json, 3 GitLab jobs), and use
pytest's `monkeypatch` to fake the download in `fetch_greenhouse`.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jobs_search.fetchers import FETCHERS, greenhouse
from jobs_search.fetchers.greenhouse import (
    description_to_text,
    fetch_greenhouse,
    parse_greenhouse,
    parse_salary,
)

# Path to the saved response. Tests are always run from the repo root.
FIXTURE = Path("tests/fixtures/greenhouse.json")


def load_fixture() -> dict:
    """Read the saved Greenhouse response and turn the JSON into Python data."""
    with open(FIXTURE, encoding="utf-8") as file:
        return json.load(file)


def test_parse_greenhouse_reads_every_job():
    jobs = parse_greenhouse(load_fixture(), company="gitlab")

    assert len(jobs) == 3


def test_parse_greenhouse_maps_fields():
    first = parse_greenhouse(load_fixture(), company="gitlab")[0]

    assert first.id == "greenhouse:gitlab:8638232002"
    assert first.source == "greenhouse"
    assert first.company == "gitlab"
    assert first.title == "AI Transformation Owner, CRO"
    assert first.location == "Remote, United States"
    assert first.url == "https://job-boards.greenhouse.io/gitlab/jobs/8638232002"
    # 13920000 cents and 23520000 cents in the fixture, as whole dollars.
    assert first.salary_min == 139200
    assert first.salary_max == 235200
    # "2026-11-06T15:25:07-05:00": 5 hours behind UTC.
    assert first.updated_at == datetime(
        2026, 11, 6, 15, 25, 7, tzinfo=timezone(timedelta(hours=-5))
    )
    # "first_published": "2026-07-22T13:38:40-04:00": 4 hours behind UTC
    # (it was summer, so daylight saving time).
    assert first.posted_at == datetime(
        2026, 7, 22, 13, 38, 40, tzinfo=timezone(timedelta(hours=-4))
    )


def test_description_is_plain_text():
    first = parse_greenhouse(load_fixture(), company="gitlab")[0]

    # No HTML tags and no escaped tags left over.
    assert "<" not in first.description
    assert "&lt;" not in first.description
    # But the actual words are still there.
    assert "GitLab is the intelligent orchestration platform" in first.description


def test_job_without_pay_has_no_salary():
    # The second job in the fixture lists no pay range.
    second = parse_greenhouse(load_fixture(), company="gitlab")[1]

    assert second.salary_min is None
    assert second.salary_max is None


def test_description_to_text_handles_escaped_html():
    escaped = "&lt;p&gt;Build &lt;b&gt;pipelines&lt;/b&gt;&lt;/p&gt;&lt;p&gt;Ship it&lt;/p&gt;"

    assert description_to_text(escaped) == "Build\npipelines\nShip it"


def test_parse_salary_ignores_other_currencies():
    # A made-up job with only a salary in euros.
    job_data = {
        "pay_input_ranges": [
            {"currency_type": "EUR", "min_cents": 5000000, "max_cents": 7000000}
        ]
    }

    assert parse_salary(job_data) == (None, None)


class FakeResponse:
    """Pretends to be the response httpx.get() returns, without any network.

    It has the two methods fetch_greenhouse() uses: raise_for_status() and json().
    """

    def __init__(self, data: dict):
        self.data = data

    def raise_for_status(self) -> None:
        pass  # pretend the request worked

    def json(self) -> dict:
        return self.data


def test_fetch_greenhouse_requests_the_right_address(monkeypatch):
    # We'll record what fetch_greenhouse() asks for here, to check it later.
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(load_fixture())

    # `monkeypatch.setattr` swaps out httpx.get for fake_get, for this test only.
    # pytest puts the real one back automatically when the test ends.
    monkeypatch.setattr(greenhouse.httpx, "get", fake_get)

    jobs = fetch_greenhouse("gitlab")

    assert len(jobs) == 3
    assert calls[0]["url"] == "https://boards-api.greenhouse.io/v1/boards/gitlab/jobs"
    assert calls[0]["params"] == {"content": "true", "pay_transparency": "true"}
    assert calls[0]["timeout"] == 30


def test_greenhouse_is_registered():
    assert FETCHERS["greenhouse"] is fetch_greenhouse

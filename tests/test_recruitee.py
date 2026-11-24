"""Tests for fetchers/recruitee.py.

Offline, like the Greenhouse tests: they read 3 real bunq jobs saved in
tests/fixtures/recruitee.json, and fake the download with `monkeypatch`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from jobs_search.fetchers import FETCHERS, recruitee
from jobs_search.fetchers.recruitee import (
    fetch_recruitee,
    parse_date,
    parse_recruitee,
    parse_salary,
)

FIXTURE = Path("tests/fixtures/recruitee.json")


def load_fixture() -> dict:
    """Read the saved Recruitee response and turn the JSON into Python data."""
    with open(FIXTURE, encoding="utf-8") as file:
        return json.load(file)


def test_parse_recruitee_reads_every_job():
    assert len(parse_recruitee(load_fixture(), company="bunq")) == 3


def test_parse_recruitee_maps_fields():
    first = parse_recruitee(load_fixture(), company="bunq")[0]

    assert first.id == "recruitee:bunq:2823928"
    assert first.source == "recruitee"
    assert first.company == "bunq"
    assert first.title == "Security Engineer"
    assert first.location == "Amsterdam, Noord-Holland, Netherlands"
    assert first.url == "https://careers.bunq.com/o/security-engineer-3"
    # bunq lists no salaries.
    assert first.salary_min is None
    assert first.salary_max is None
    # "published_at": "2026-11-11 13:21:40 UTC"
    assert first.posted_at == datetime(2026, 11, 11, 13, 21, 40, tzinfo=UTC)
    # "updated_at": "2026-11-19 10:47:03 UTC"
    assert first.updated_at == datetime(2026, 11, 19, 10, 47, 3, tzinfo=UTC)


def test_remote_job_keeps_its_country():
    # The second job is remote. Its "location" is only "Remote job".
    second = parse_recruitee(load_fixture(), company="bunq")[1]

    assert second.location == "Remote job; France"


def test_description_joins_both_parts_as_plain_text():
    first = parse_recruitee(load_fixture(), company="bunq")[0]

    assert "<" not in first.description
    # From the "description" part...
    assert "we’re on a mission to Make Life Easy" in first.description
    # ...and from the "requirements" part.
    assert "You write code to automate security work" in first.description


def test_unpublished_offers_are_skipped():
    data = load_fixture()
    data["offers"][0]["status"] = "closed"

    assert len(parse_recruitee(data, company="bunq")) == 2


def test_parse_date():
    assert parse_date("2026-10-22 06:17:49 UTC") == "2026-10-22 06:17:49+00:00"
    assert parse_date(None) is None


def test_parse_salary_keeps_only_yearly_us_dollars():
    yearly_usd = {
        "salary": {
            "min": "100000",
            "max": "120000",
            "period": "year",
            "currency": "USD",
        }
    }
    yearly_eur = {
        "salary": {
            "min": "100000",
            "max": "120000",
            "period": "year",
            "currency": "EUR",
        }
    }
    monthly_usd = {
        "salary": {"min": "4000", "max": "5700", "period": "month", "currency": "USD"}
    }

    assert parse_salary(yearly_usd) == (100000, 120000)
    assert parse_salary(yearly_eur) == (None, None)
    assert parse_salary(monthly_usd) == (None, None)
    assert parse_salary({}) == (None, None)


class FakeResponse:
    """Pretends to be the response httpx.get() returns, without any network."""

    def __init__(self, data: dict):
        self.data = data

    def raise_for_status(self) -> None:
        pass  # pretend the request worked

    def json(self) -> dict:
        return self.data


def test_fetch_recruitee_requests_the_right_address(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append({"url": url, "timeout": timeout})
        return FakeResponse(load_fixture())

    monkeypatch.setattr(recruitee.httpx, "get", fake_get)

    jobs = fetch_recruitee("bunq")

    assert len(jobs) == 3
    assert calls[0]["url"] == "https://bunq.recruitee.com/api/offers/"
    assert calls[0]["timeout"] == 30


def test_recruitee_is_registered():
    assert FETCHERS["recruitee"] is fetch_recruitee

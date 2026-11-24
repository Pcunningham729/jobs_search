"""Tests for fetchers/personio.py.

Offline, like the Greenhouse tests: they read 3 real ottonova jobs saved in
tests/fixtures/personio.xml, and fake the download with `monkeypatch`.
"""

from datetime import UTC, datetime
from pathlib import Path

from jobs_search.fetchers import FETCHERS, personio
from jobs_search.fetchers.personio import fetch_personio, parse_personio

FIXTURE = Path("tests/fixtures/personio.xml")


def load_fixture() -> str:
    """Read the saved Personio feed. It's XML, so we keep it as text."""
    return FIXTURE.read_text(encoding="utf-8")


def test_parse_personio_reads_every_job():
    assert len(parse_personio(load_fixture(), company="ottonova")) == 3


def test_parse_personio_maps_fields():
    first = parse_personio(load_fixture(), company="ottonova")[0]

    assert first.id == "personio:ottonova:2888695"
    assert first.source == "personio"
    assert first.company == "ottonova"
    assert first.title == (
        "(Junior) Customer Support Agent - private Krankenversicherung (m/w/d)"
    )
    # The main <office>, then the one in <additionalOffices>.
    assert first.location == "München; Köln"
    # Built from the id, since the feed has no link.
    assert first.url == "https://ottonova.jobs.personio.de/job/2888695"
    assert first.salary_min is None
    assert first.salary_max is None
    # "<createdAt>2026-11-13T16:21:35+00:00</createdAt>"
    assert first.posted_at == datetime(2026, 11, 13, 16, 21, 35, tzinfo=UTC)
    assert first.updated_at is None


def test_job_without_sections_has_empty_description():
    # The first job's <jobDescriptions> is empty, as is common on Personio.
    first = parse_personio(load_fixture(), company="ottonova")[0]

    assert first.description == ""


def test_description_sections_become_plain_text():
    third = parse_personio(load_fixture(), company="ottonova")[2]

    assert "<" not in third.description
    # Each section's heading comes first, then its text.
    assert third.description.startswith("YOUR TEAM\nottonova is Germany")
    assert "YOUR SKILLSET" in third.description


def test_xml_escapes_are_undone():
    # In the feed the title is written "Architecture &amp; Platform".
    third = parse_personio(load_fixture(), company="ottonova")[2]

    assert "Architecture & Platform" in third.title


class FakeResponse:
    """Pretends to be the response httpx.get() returns, without any network.

    Personio sends XML, so fetch_personio() reads `.text` instead of `.json()`.
    """

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        pass  # pretend the request worked


def test_fetch_personio_requests_the_right_address(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(load_fixture())

    monkeypatch.setattr(personio.httpx, "get", fake_get)

    jobs = fetch_personio("ottonova")

    assert len(jobs) == 3
    assert calls[0]["url"] == "https://ottonova.jobs.personio.de/xml"
    assert calls[0]["params"] == {"language": "en"}
    assert calls[0]["timeout"] == 30


def test_personio_is_registered():
    assert FETCHERS["personio"] is fetch_personio

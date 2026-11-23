"""Tests for main.py: the command line reads options and runs the right step.

Environment variables point every path at tmp_path, and the real embedder
is swapped for a fake, so nothing touches your data, the network or the model.
"""

import shutil
from pathlib import Path

import pytest

from jobs_search import main as cli
from jobs_search.discovery import Company, load_companies, save_companies
from jobs_search.fetchers import FETCHERS
from jobs_search.models import Job

FIXTURE_PROFILE = Path(__file__).parent / "fixtures" / "profile"


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(text.lower().count("data")), 1.0] for text in texts]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A throwaway copy of everything the CLI reads and writes."""
    shutil.copytree(FIXTURE_PROFILE, tmp_path / "profile")
    monkeypatch.setenv("JOBS_SEARCH_PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("JOBS_SEARCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JOBS_SEARCH_COMPANIES_FILE", str(tmp_path / "companies.csv"))
    monkeypatch.setattr(cli, "embed_texts", fake_embed)
    return tmp_path


def test_parser_reads_options():
    args = cli.build_parser().parse_args(["match", "--title", "analyst", "--days", "7"])

    assert (args.title, args.days, args.limit) == ("analyst", 7, 10)
    assert args.run is cli.run_match


def test_fetch_without_companies_says_what_to_do(workspace, capsys):
    cli.main(["fetch"])

    assert "Run: uv run jobs-search discover" in capsys.readouterr().out


def test_discover_with_seeds_only(workspace, monkeypatch, capsys):
    (workspace / "company_seeds.txt").write_text("lever acme\n", encoding="utf-8")
    monkeypatch.setattr("jobs_search.discovery.board_exists", lambda company: True)
    monkeypatch.setattr("jobs_search.discovery.time.sleep", lambda seconds: None)

    cli.main(["discover", "--skip-crawl"])

    assert load_companies(workspace / "companies.csv") == [
        Company(source="lever", slug="acme")
    ]


def test_fetch_then_match(workspace, monkeypatch, capsys):
    save_companies(workspace / "companies.csv", [Company(source="lever", slug="acme")])
    job = Job(
        id="lever:acme:1", source="lever", company="acme", title="Data Engineer",
        location="Remote (US)", url="https://example.com/1",
        description="Data pipelines.",
    )  # fmt: skip
    # Swap the real Lever fetcher for one that returns our test job.
    monkeypatch.setitem(FETCHERS, "lever", lambda slug: [job])

    cli.main(["fetch"])
    cli.main(["match", "--limit", "5"])

    out = capsys.readouterr().out
    assert "1 jobs, 1 new" in out
    assert "1. Data Engineer at acme" in out

"""Tests for discovery.py. No network: checks use a fake, files use tmp_path."""

import httpx
import pytest

from jobs_search.discovery import (
    Company,
    company_from_url,
    discover,
    load_companies,
    parse_index_lines,
    remove_companies,
    save_companies,
    seed_candidates,
)


def lever(slug: str) -> Company:
    return Company(source="lever", slug=slug)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://jobs.lever.co/spotify/1a2b-3c4d", lever("spotify")),
        ("https://jobs.lever.co/Spotify", lever("spotify")),  # lowercased
        ("https://boards.greenhouse.io/gitlab/jobs/123",
         Company(source="greenhouse", slug="gitlab")),
        ("https://job-boards.greenhouse.io/gitlab",
         Company(source="greenhouse", slug="gitlab")),
        ("https://boards.greenhouse.io/embed/job_board?for=gitlab",
         Company(source="greenhouse", slug="gitlab")),
        ("https://jobs.ashbyhq.com/ramp/abc", Company(source="ashby", slug="ramp")),
        ("https://jobs.lever.co/", None),  # no company in the address
        ("https://jobs.lever.co/robots.txt", None),  # a page, not a company
        ("https://example.com/spotify", None),  # not a job board
    ],
)  # fmt: skip
def test_company_from_url(url, expected):
    assert company_from_url(url) == expected


def test_parse_index_lines_skips_junk_and_duplicates():
    text = (
        '{"url": "https://jobs.lever.co/spotify/1"}\n'
        '{"url": "https://jobs.lever.co/spotify/2"}\n'
        "<html>error page</html>\n"
        '{"url": "https://jobs.lever.co/plaid/9"}\n'
    )

    assert parse_index_lines(text) == {lever("spotify"), lever("plaid")}


def test_seed_candidates(tmp_path):
    path = tmp_path / "seeds.txt"
    path.write_text(
        "# a comment\n\nlever spotify   # trailing comment\nnotaboard acme\n"
        "too many words here\n",
        encoding="utf-8",
    )

    assert seed_candidates(path) == {lever("spotify")}
    assert seed_candidates(tmp_path / "missing.txt") == set()


def test_companies_file_round_trip_sorted_and_unique(tmp_path):
    path = tmp_path / "companies.csv"
    save_companies(path, [lever("zeta"), lever("alpha"), lever("zeta")])

    assert load_companies(path) == [lever("alpha"), lever("zeta")]
    assert load_companies(tmp_path / "missing.csv") == []


def test_discover_adds_real_boards_and_only_checks_new_ones(tmp_path):
    path = tmp_path / "companies.csv"
    save_companies(path, [lever("known")])
    checked = []

    def fake_check(company: Company) -> bool:
        checked.append(company.slug)
        if company.slug == "flaky":
            raise httpx.ConnectError("no network")
        return company.slug == "real"

    candidates = {lever("known"), lever("real"), lever("fake"), lever("flaky")}
    report = discover(path, candidates, check=fake_check, pause_seconds=0)

    assert "known" not in checked  # already on the list: not checked again
    assert report.added == [lever("real")]
    assert (report.already_known, report.not_found, report.errors) == (1, 1, 1)
    assert load_companies(path) == [lever("known"), lever("real")]


def test_remove_companies(tmp_path):
    path = tmp_path / "companies.csv"
    save_companies(path, [lever("a"), lever("b")])

    remove_companies(path, [lever("a")])

    assert load_companies(path) == [lever("b")]


def test_real_seed_file_parses():
    from pathlib import Path

    seeds = seed_candidates(Path(__file__).parent.parent / "company_seeds.txt")

    assert lever("spotify") in seeds
    assert len(seeds) > 50

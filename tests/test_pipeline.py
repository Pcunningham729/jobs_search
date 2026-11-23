"""Tests for pipeline.py, with fake fetchers (no network) and a fake embedder."""

import httpx
import pytest

from jobs_search.discovery import Company, load_companies, save_companies
from jobs_search.models import Job
from jobs_search.pipeline import fetch_and_index
from jobs_search.profile import Filters, Interest, Profile
from jobs_search.rag import get_title_vector, open_index
from jobs_search.store import connect, get_job

PROFILE = Profile(
    titles=["data engineer"],
    interests=[Interest(name="pipelines", text="data pipeline")],
    filters=Filters(locations=["Minneapolis"], allow_remote=False),
)


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(text.lower().count("data")), 1.0] for text in texts]


def make_job(job_id: str, location: str, description: str = "Data work.") -> Job:
    return Job(
        id=job_id, source="lever", company="acme", title="Data Engineer",
        location=location, url="https://example.com", description=description,
    )  # fmt: skip


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def fake_lever(slug: str) -> list[Job]:
    if slug == "closed":
        raise http_error(404)
    if slug == "broken":
        raise http_error(500)
    return [
        make_job("lever:acme:mpls", "Minneapolis, MN"),
        make_job("lever:acme:paris", "Paris"),
    ]


@pytest.fixture
def setup(tmp_path):
    conn = connect(tmp_path / "jobs.db")
    index = open_index(tmp_path / "chroma")
    yield conn, index, tmp_path
    conn.close()


def run(conn, index, companies, companies_file=None, fetcher=fake_lever):
    return fetch_and_index(
        conn, index, companies, PROFILE, fake_embed,
        companies_file=companies_file, fetchers={"lever": fetcher},
    )  # fmt: skip


def test_saves_every_job_but_indexes_only_those_passing_filters(setup):
    conn, index, _ = setup

    report = run(conn, index, [Company(source="lever", slug="acme")])

    assert (report.jobs_seen, report.new_jobs, report.indexed_jobs) == (2, 2, 1)
    assert get_job(conn, "lever:acme:paris") is not None  # saved...
    assert get_title_vector(index, "lever:acme:paris") is None  # ...not indexed
    assert get_title_vector(index, "lever:acme:mpls") is not None


def test_second_run_indexes_only_changed_jobs(setup):
    conn, index, _ = setup
    acme = [Company(source="lever", slug="acme")]
    run(conn, index, acme)

    assert run(conn, index, acme).indexed_jobs == 0  # nothing changed

    def changed_lever(slug: str) -> list[Job]:
        return [make_job("lever:acme:mpls", "Minneapolis, MN", "New data text.")]

    report = run(conn, index, acme, fetcher=changed_lever)
    assert (report.changed_jobs, report.indexed_jobs) == (1, 1)


def test_failures_dont_stop_the_run_and_closed_boards_are_removed(setup):
    conn, index, tmp_path = setup
    companies = [
        Company(source="lever", slug="closed"),
        Company(source="lever", slug="broken"),
        Company(source="lever", slug="acme"),
    ]
    path = tmp_path / "companies.csv"
    save_companies(path, companies)

    report = run(conn, index, companies, companies_file=path)

    assert report.companies_fetched == 1  # acme still got fetched
    assert report.failed == ["lever:broken"]
    assert report.removed == ["lever:closed"]
    assert Company(source="lever", slug="closed") not in load_companies(path)
    assert Company(source="lever", slug="broken") in load_companies(path)

"""Tests for store.py.

Each test gets a brand-new, empty database in a temporary folder, thanks to
pytest's `tmp_path`: a folder pytest creates for one test and cleans up
later. So tests never touch your real data/jobs.db, and never affect each other.
"""

from pathlib import Path

import pytest

from jobs_search.models import Job
from jobs_search.store import (
    connect,
    content_hash,
    database_path,
    get_job,
    get_status,
    list_jobs,
    save_jobs,
    set_status,
)


def make_job(job_id: str = "lever:acme:1", **changes) -> Job:
    """Build a test job. Pass field names to change any of them.

    `**changes` collects extra named arguments into a dict, so
    make_job(title="Analyst") gives a job whose title is "Analyst".
    """
    fields = {
        "id": job_id,
        "source": "lever",
        "company": "acme",
        "title": "Data Engineer",
        "location": "Minneapolis, MN",
        "url": "https://jobs.lever.co/acme/1",
        "description": "Build pipelines.",
    }
    fields.update(changes)  # overwrite any fields the test asked to change
    return Job(**fields)


@pytest.fixture
def conn(tmp_path):
    """A fresh database for each test.

    A pytest "fixture" is setup code that tests ask for by name: any test
    with a `conn` argument gets this connection automatically.
    """
    connection = connect(tmp_path / "test.db")
    yield connection  # the test runs here
    connection.close()  # then this runs, after the test finishes


def test_connect_creates_the_file_and_folder(tmp_path):
    path = tmp_path / "some" / "folder" / "jobs.db"

    connect(path).close()

    assert path.exists()


def test_database_path_reads_the_environment_variable(monkeypatch):
    monkeypatch.setenv("JOBS_SEARCH_DATA_DIR", "somewhere/else")

    assert database_path() == Path("somewhere/else/jobs.db")


def test_database_path_defaults_to_data_folder(monkeypatch):
    monkeypatch.delenv("JOBS_SEARCH_DATA_DIR", raising=False)

    assert database_path() == Path("data/jobs.db")


def test_new_jobs_are_saved_and_reported(conn):
    result = save_jobs(conn, [make_job("lever:acme:1"), make_job("lever:acme:2")])

    assert result.new_ids == ["lever:acme:1", "lever:acme:2"]
    assert result.changed_ids == []
    assert len(list_jobs(conn)) == 2


def test_saving_the_same_job_twice_does_not_duplicate_it(conn):
    save_jobs(conn, [make_job()])
    result = save_jobs(conn, [make_job()])

    assert result.new_ids == []
    assert result.unchanged_count == 1
    assert len(list_jobs(conn)) == 1


def test_changed_job_is_detected_and_updated(conn):
    save_jobs(conn, [make_job(description="Build pipelines.")])
    result = save_jobs(conn, [make_job(description="Build pipelines and dashboards.")])

    assert result.changed_ids == ["lever:acme:1"]
    assert (
        get_job(conn, "lever:acme:1").description == "Build pipelines and dashboards."
    )


def test_get_job_returns_the_same_job(conn):
    job = make_job(salary_min=100000)
    save_jobs(conn, [job])

    # The job read back from the database equals the one we saved.
    assert get_job(conn, job.id) == job


def test_get_job_returns_none_for_unknown_id(conn):
    assert get_job(conn, "lever:acme:does-not-exist") is None


def test_new_jobs_start_with_status_new(conn):
    save_jobs(conn, [make_job()])

    assert get_status(conn, "lever:acme:1") == "new"


def test_set_status_and_filter_by_status(conn):
    save_jobs(conn, [make_job("lever:acme:1"), make_job("lever:acme:2")])

    set_status(conn, "lever:acme:2", "hidden")

    new_jobs = list_jobs(conn, statuses=["new"])
    assert [job.id for job in new_jobs] == ["lever:acme:1"]


def test_status_is_kept_when_a_job_changes(conn):
    save_jobs(conn, [make_job(description="Old text.")])
    set_status(conn, "lever:acme:1", "rejected")

    save_jobs(conn, [make_job(description="New text.")])

    assert get_status(conn, "lever:acme:1") == "rejected"


def test_set_status_rejects_unknown_status(conn):
    save_jobs(conn, [make_job()])

    with pytest.raises(ValueError):
        set_status(conn, "lever:acme:1", "maybe")


def test_set_status_rejects_unknown_job(conn):
    with pytest.raises(ValueError):
        set_status(conn, "lever:acme:does-not-exist", "applied")


def test_hash_ignores_dates_but_not_words():
    original = make_job()
    with_date = make_job(updated_at="2026-11-20T09:30:00")
    with_new_title = make_job(title="Senior Data Engineer")

    assert content_hash(original) == content_hash(with_date)
    assert content_hash(original) != content_hash(with_new_title)

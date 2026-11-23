"""The fetch step: download every company's jobs, save them, index the useful ones.

This is the "collect" half of the project, joining modules that already
exist:

    companies.csv --> fetchers/ --> list[Job] --> store.save_jobs()   (SQLite)
                                                 |
                       jobs passing your filters  +--> rag.index_jobs() (Chroma)

Only jobs that pass your location and salary filters get embedded and
indexed. matching.rank_jobs() skips the others anyway, and embedding is the
slow part, so this saves most of the time. Every job is still saved in
SQLite (the source of truth), so if you change your filters, the next fetch
indexes the newly-passing jobs.

Run it with:  uv run jobs-search fetch
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel

from jobs_search.discovery import Company, remove_companies
from jobs_search.embeddings import Embedder
from jobs_search.fetchers import FETCHERS
from jobs_search.matching import passes_filters
from jobs_search.models import Job
from jobs_search.profile import Profile
from jobs_search.rag import JobIndex, get_title_vector, index_jobs
from jobs_search.store import save_jobs

# A fetcher takes a company slug and returns its jobs, like fetch_lever("spotify").
Fetcher = Callable[[str], list[Job]]


class FetchReport(BaseModel):
    """What one fetch run did."""

    companies_fetched: int = 0
    jobs_seen: int = 0
    new_jobs: int = 0
    changed_jobs: int = 0
    indexed_jobs: int = 0
    failed: list[str] = []  # "lever:acme" that errored; tried again next time
    removed: list[str] = []  # boards that answered 404: taken off the list


def jobs_to_index(
    jobs: list[Job], changed_ids: set[str], index: JobIndex, profile: Profile
) -> list[Job]:
    """Pick the jobs worth embedding: they pass the filters, and are new,
    changed, or somehow missing from the index."""
    picked = []
    for job in jobs:
        if not passes_filters(job, profile.filters):
            continue
        if job.id in changed_ids or get_title_vector(index, job.id) is None:
            picked.append(job)
    return picked


def fetch_and_index(
    conn: sqlite3.Connection,
    index: JobIndex,
    companies: list[Company],
    profile: Profile,
    embed: Embedder,
    companies_file: Path | None = None,
    fetchers: dict[str, Fetcher] = FETCHERS,
) -> FetchReport:
    """Fetch, save and index jobs for every company on the list.

    One company failing never stops the run: its error is counted and the
    loop moves on. Boards that answer 404 (closed) are removed from
    `companies_file`, if one is given. `fetchers` is passed in by tests.
    """
    report = FetchReport()
    gone = []

    for number, company in enumerate(companies, start=1):
        name = f"{company.source}:{company.slug}"
        print(f"[{number}/{len(companies)}] {name}", end=" ", flush=True)
        try:
            jobs = fetchers[company.source](company.slug)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                print("board is gone; removing it")
                gone.append(company)
                report.removed.append(name)
            else:
                print(f"failed ({error.response.status_code})")
                report.failed.append(name)
            continue
        except httpx.HTTPError as error:  # timeouts, no connection, ...
            print(f"failed ({type(error).__name__})")
            report.failed.append(name)
            continue

        saved = save_jobs(conn, jobs)
        changed_ids = set(saved.new_ids) | set(saved.changed_ids)  # | joins sets
        to_index = jobs_to_index(jobs, changed_ids, index, profile)
        index_jobs(index, to_index, embed)

        print(
            f"{len(jobs)} jobs, {len(saved.new_ids)} new, "
            f"{len(saved.changed_ids)} changed, {len(to_index)} indexed"
        )
        report.companies_fetched += 1
        report.jobs_seen += len(jobs)
        report.new_jobs += len(saved.new_ids)
        report.changed_jobs += len(saved.changed_ids)
        report.indexed_jobs += len(to_index)

    if gone and companies_file is not None:
        remove_companies(companies_file, gone)
    return report

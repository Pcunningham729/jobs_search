"""Save jobs in SQLite, the project's source of truth.

SQLite is a database that lives in a single file (data/jobs.db). It comes
built into Python as the `sqlite3` module, so there's nothing to install.

What this module does:
- Saves jobs without duplicates. `Job.id` is the key, so saving the same job
  twice updates it instead of adding a second copy.
- Notices changed postings. Each job gets a "hash": a short fingerprint
  calculated from its text. If any word changes, the fingerprint changes.
- Keeps your status for each job: new, applied, rejected or hidden.

Every other part of the program reads jobs from here. The Chroma search
index can always be rebuilt from this database.

How to use it:

    conn = connect(database_path())
    result = save_jobs(conn, jobs)      # jobs is a list[Job] from a fetcher
    print(result.new_ids)               # ids of jobs we'd never seen before
    set_status(conn, "lever:spotify:abc", "applied")
"""

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from jobs_search.models import Job

# The statuses a job can have. Every job starts as "new".
Status = Literal["new", "applied", "rejected", "hidden"]

# SQL is the language databases understand. This command creates the `jobs`
# table (like a spreadsheet tab) if it doesn't exist yet. Each line is a
# column: its name, its type, and any rules.
#   PRIMARY KEY  - the id must be unique; SQLite refuses a second row with it
#   NOT NULL     - the column must always have a value
#   DEFAULT 'new' - the value to use when none is given
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    company       TEXT NOT NULL,
    title         TEXT NOT NULL,
    data          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'new',
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
)
"""
# Why both columns AND a `data` column? `data` holds the whole Job as JSON
# text, so we can rebuild the exact Job later, and adding a field to Job
# doesn't require changing this table. The separate columns (source,
# company, title) are copies of a few fields, so they're easy to read when
# you open the database in a viewer.


class SaveResult(BaseModel):
    """What happened when a batch of jobs was saved.

    The ids tell the next stage (the search index) exactly which jobs need
    their text embedded: new ones and changed ones. Unchanged ones don't.
    """

    new_ids: list[str] = []
    changed_ids: list[str] = []
    unchanged_count: int = 0


def database_path() -> Path:
    """Where the database file lives.

    The folder comes from the JOBS_SEARCH_DATA_DIR environment variable (a
    setting passed in from outside the program), and defaults to `data` in
    the repo root. That way it can be moved (for example in Docker) without
    changing any code.
    """
    data_dir = os.environ.get("JOBS_SEARCH_DATA_DIR", "data")
    return Path(data_dir) / "jobs.db"


def connect(path: Path) -> sqlite3.Connection:
    """Open the database at `path`, creating the file and table if needed.

    Returns a "connection": the object you pass to every other function here
    to talk to the database.
    """
    # Make the folder (e.g. `data/`) if it doesn't exist yet.
    # `parents=True` also makes any missing folders above it;
    # `exist_ok=True` means "no error if it already exists".
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)

    # Normally SQLite gives back each row as a tuple: ("lever:...", "lever", ...).
    # `sqlite3.Row` lets us use column names instead: row["status"].
    conn.row_factory = sqlite3.Row

    conn.execute(CREATE_TABLE_SQL)
    conn.commit()  # `commit` saves changes to the file
    return conn


def content_hash(job: Job) -> str:
    """Make a fingerprint of the parts of a job that matter to you.

    If the title, location, description or salary changes, the fingerprint
    changes. Dates are left out on purpose: a board can touch `updated_at`
    without changing any words, and that shouldn't count as a change.
    """
    important_parts = {
        "title": job.title,
        "location": job.location,
        "description": job.description,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
    }
    # Turn the dict into one string. `sort_keys=True` keeps the order the
    # same every time, so the same job always gives the same string.
    text = json.dumps(important_parts, sort_keys=True)

    # SHA-256 turns any text into a 64-character fingerprint. The same text
    # always gives the same fingerprint; any change gives a different one.
    # `.encode()` turns the text into bytes, which hashlib needs.
    return hashlib.sha256(text.encode()).hexdigest()


def save_jobs(conn: sqlite3.Connection, jobs: list[Job]) -> SaveResult:
    """Save a batch of jobs, and report which were new, changed or unchanged.

    - New job: added with status "new".
    - Changed job: its data and fingerprint are updated. Your status is kept,
      so a job you rejected stays rejected even if the company edits it.
    - Unchanged job: only `last_seen_at` is updated, to record that the job
      is still open.
    """
    result = SaveResult()
    now = datetime.now(UTC).isoformat()  # e.g. "2026-11-22T19:05:00+00:00"

    for job in jobs:
        new_hash = content_hash(job)
        # `model_dump_json()` is Pydantic's way to turn a Job into JSON text.
        job_json = job.model_dump_json()

        # Look for a job with this id already in the database. The `?` is a
        # placeholder that SQLite fills in with `job.id`. Always use `?`
        # rather than pasting values into the SQL text yourself: it handles
        # quotes in the value safely.
        existing = conn.execute(
            "SELECT content_hash FROM jobs WHERE id = ?", (job.id,)
        ).fetchone()  # one row, or None if there's no match

        if existing is None:
            conn.execute(
                """
                INSERT INTO jobs (id, source, company, title, data,
                                  content_hash, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.source,
                    job.company,
                    job.title,
                    job_json,
                    new_hash,
                    now,
                    now,
                ),
            )
            result.new_ids.append(job.id)

        elif existing["content_hash"] != new_hash:
            conn.execute(
                """
                UPDATE jobs
                SET title = ?, data = ?, content_hash = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (job.title, job_json, new_hash, now, job.id),
            )
            result.changed_ids.append(job.id)

        else:
            conn.execute("UPDATE jobs SET last_seen_at = ? WHERE id = ?", (now, job.id))
            result.unchanged_count += 1

    conn.commit()
    return result


def get_job(conn: sqlite3.Connection, job_id: str) -> Job | None:
    """Return one job by its id, or None if it isn't in the database."""
    row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    # The reverse of model_dump_json(): JSON text back into a Job.
    return Job.model_validate_json(row["data"])


def list_jobs(
    conn: sqlite3.Connection, statuses: list[Status] | None = None
) -> list[Job]:
    """Return saved jobs, optionally only those with certain statuses.

    Examples:
        list_jobs(conn)                    # every job
        list_jobs(conn, statuses=["new"])  # only jobs you haven't acted on
    """
    if statuses is None:
        rows = conn.execute("SELECT data FROM jobs ORDER BY id").fetchall()
    else:
        # Build one `?` per status: ["new", "applied"] -> "?, ?"
        placeholders = ", ".join("?" for _ in statuses)
        rows = conn.execute(
            f"SELECT data FROM jobs WHERE status IN ({placeholders}) ORDER BY id",
            statuses,
        ).fetchall()

    jobs = []
    for row in rows:
        jobs.append(Job.model_validate_json(row["data"]))
    return jobs


def get_status(conn: sqlite3.Connection, job_id: str) -> Status | None:
    """Return a job's status, or None if the job isn't in the database."""
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return row["status"]


def set_status(conn: sqlite3.Connection, job_id: str, status: Status) -> None:
    """Change a job's status, for example to "applied" or "hidden".

    Raises ValueError if the status isn't one of the allowed ones, or if
    there's no job with that id, so a typo never fails silently.
    """
    # Type hints don't stop wrong values while the program runs, so check here.
    allowed = ["new", "applied", "rejected", "hidden"]
    if status not in allowed:
        raise ValueError(f"Unknown status {status!r}. Use one of: {allowed}")

    cursor = conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()

    # `rowcount` is how many rows the UPDATE changed. 0 means no job had that id.
    if cursor.rowcount == 0:
        raise ValueError(f"No job with id {job_id!r}")

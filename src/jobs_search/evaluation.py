"""Measure how well matching.py ranks jobs, using jobs you've labeled by hand.

You label jobs "good" or "bad" in profile/labeled_jobs.csv. The eval then
scores just those jobs with the current profile, weights, chunking and
model, and asks: does the ranking put your good jobs first?

The main number is precision@10 (say "precision at ten"): of the 10 labeled
jobs ranked highest, what share did you label good? 1.0 is perfect; 0.5
means half of the top 10 were jobs you don't want.

Per CLAUDE.md: run the eval before AND after any change to scoring,
chunking, the embedding model, the profile format or agent_prompts/, and
compare the two numbers.

main.py runs it with two commands:
    uv run jobs-search label   -> a sheet of jobs to label
    uv run jobs-search eval    -> score the ranking
"""

import csv
import random
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from jobs_search.embeddings import Embedder
from jobs_search.matching import embed_profile, passes_filters, rank_jobs, score_job
from jobs_search.models import Job
from jobs_search.profile import Profile
from jobs_search.rag import JobIndex, get_job_vectors, get_title_vector
from jobs_search.store import get_job, list_jobs

Label = Literal["good", "bad"]

# The columns of the label sheet, in order. You only fill in `label` (and
# `note` if you like); the rest help you recognise the job.
SHEET_COLUMNS = ["job_id", "label", "note", "title", "company", "location", "url"]


class LabeledJob(BaseModel):
    """One row of labeled_jobs.csv that you've labeled."""

    job_id: str
    label: Label
    note: str = ""


class ScoredLabel(BaseModel):
    """A labeled job, with the score the current ranking gives it."""

    job_id: str
    title: str
    label: Label
    score: float
    passes_filters: bool  # False means the filters would hide it completely


class EvalReport(BaseModel):
    """Everything one eval run found."""

    precision_at_10: float
    good_count: int
    bad_count: int
    ranked: list[ScoredLabel]  # every labeled job, best score first
    # Good jobs with at least one bad job ranked above them: title -> how many.
    good_below_bad: dict[str, int]
    # Good jobs that the location/salary filters would drop entirely.
    good_filtered_out: list[str]
    # Labeled ids that aren't in the store or index (e.g. a typo in the id).
    missing_ids: list[str]


# ---------------------------------------------------------------------------
# Reading and writing CSV files
# ---------------------------------------------------------------------------


def load_labels(path: Path) -> list[LabeledJob]:
    """Read labeled_jobs.csv. Rows with an empty label are skipped (not labeled yet).

    Labels are case-insensitive, so "Good" and "GOOD" work too.

    Raises:
        ValueError: for a label that isn't good or bad, naming the line.
    """
    labels = []
    # "utf-8-sig" also handles the invisible marker Excel adds to the start of
    # CSV files it saves. `newline=""` is what Python's csv module asks for.
    with open(path, encoding="utf-8-sig", newline="") as file:
        # DictReader turns each row into a dict, using the header row as keys:
        # {"job_id": "lever:...", "label": "good", ...}
        reader = csv.DictReader(file)
        # `enumerate(..., start=2)` counts rows as we go. Line 1 is the header,
        # so the first data row is line 2: matches what Excel shows you.
        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip().lower()
            if not label:
                continue
            if label not in ("good", "bad"):
                raise ValueError(
                    f"{path.name} line {line_number}: label {label!r} "
                    "should be 'good' or 'bad' (or left empty)."
                )
            labels.append(
                LabeledJob(
                    job_id=row["job_id"].strip(),
                    label=label,
                    note=(row.get("note") or "").strip(),
                )
            )
    return labels


def write_label_sheet(path: Path, jobs: list[Job]) -> None:
    """Write a CSV of jobs to label, with empty `label` and `note` columns.

    Refuses to overwrite an existing file, so your labels are never lost.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Rename or delete it to make a new sheet."
        )
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_id": job.id,
                    "label": "",
                    "note": "",
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "url": job.url,
                }
            )


def last_precision(history_path: Path) -> float | None:
    """The precision@10 from the most recent saved run, or None if there isn't one."""
    if not history_path.exists():
        return None
    with open(history_path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    return float(rows[-1]["precision_at_10"])  # [-1] is the last item


def append_history(history_path: Path, report: EvalReport, note: str = "") -> None:
    """Add this run's numbers as a new row at the end of the history file."""
    columns = ["run_at", "precision_at_10", "good", "bad", "good_below_bad", "note"]
    is_new_file = not history_path.exists()
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # "a" means append: add to the end instead of replacing the file.
    with open(history_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if is_new_file:
            writer.writeheader()
        writer.writerow(
            {
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "precision_at_10": f"{report.precision_at_10:.3f}",
                "good": report.good_count,
                "bad": report.bad_count,
                "good_below_bad": len(report.good_below_bad),
                "note": note,
            }
        )


# ---------------------------------------------------------------------------
# Choosing jobs to label
# ---------------------------------------------------------------------------


def pick_jobs_to_label(
    conn: sqlite3.Connection,
    index: JobIndex,
    profile: Profile,
    embed: Embedder,
    top_count: int = 15,
    random_count: int = 15,
    seed: int = 0,
) -> list[Job]:
    """Pick jobs for you to label: the current top-ranked ones, plus random ones.

    Only labeling the ranker's favourites would hide what it's missing, so
    half the sheet is random jobs that pass your filters. `seed` makes the
    random choice the same every time (handy for tests).
    """
    top = [
        result.job for result in rank_jobs(conn, index, profile, embed, limit=top_count)
    ]
    top_ids = {job.id for job in top}  # a set: fast "is this id in here?" checks

    others = []
    for job in list_jobs(conn):
        if job.id not in top_ids and passes_filters(job, profile.filters):
            others.append(job)

    chooser = random.Random(seed)
    sample = chooser.sample(others, min(random_count, len(others)))

    # Shuffle the two groups together, so you can't tell which is which
    # while labeling (that keeps your labels honest).
    picked = top + sample
    chooser.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# Scoring the labeled jobs
# ---------------------------------------------------------------------------


def precision_at_k(ranked: list[ScoredLabel], k: int = 10) -> float:
    """The share of the top `k` ranked jobs that are labeled good."""
    top = ranked[:k]
    if not top:
        return 0.0
    good = 0
    for item in top:
        if item.label == "good":
            good += 1
    return good / len(top)


def find_good_below_bad(ranked: list[ScoredLabel]) -> dict[str, int]:
    """For each good job, count the bad jobs ranked above it (only if any are)."""
    result = {}
    bad_seen = 0  # bad jobs seen so far, walking down from the top
    for item in ranked:
        if item.label == "bad":
            bad_seen += 1
        elif bad_seen > 0:
            result[item.title] = bad_seen
    return result


def evaluate(
    conn: sqlite3.Connection,
    index: JobIndex,
    profile: Profile,
    embed: Embedder,
    labels: list[LabeledJob],
) -> EvalReport:
    """Score every labeled job with the current settings and build a report.

    Unlike rank_jobs(), this ignores the filters and statuses when ranking,
    so it measures the SCORING. It reports separately any good jobs the
    filters would drop, because that's a different kind of mistake.
    """
    profile_vectors = embed_profile(profile, embed)

    ranked = []
    missing_ids = []
    for labeled in labels:
        job = get_job(conn, labeled.job_id)
        title_vector = get_title_vector(index, labeled.job_id)
        chunk_vectors = get_job_vectors(index, labeled.job_id)
        if job is None or title_vector is None or not chunk_vectors:
            missing_ids.append(labeled.job_id)
            continue

        scores = score_job(
            title_vector, chunk_vectors, profile_vectors, profile.weights
        )
        ranked.append(
            ScoredLabel(
                job_id=job.id,
                title=job.title,
                label=labeled.label,
                score=scores.total,
                passes_filters=passes_filters(job, profile.filters),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)

    good_filtered_out = []
    for item in ranked:
        if item.label == "good" and not item.passes_filters:
            good_filtered_out.append(item.title)

    return EvalReport(
        precision_at_10=precision_at_k(ranked, 10),
        good_count=sum(1 for item in ranked if item.label == "good"),
        bad_count=sum(1 for item in ranked if item.label == "bad"),
        ranked=ranked,
        good_below_bad=find_good_below_bad(ranked),
        good_filtered_out=good_filtered_out,
        missing_ids=missing_ids,
    )

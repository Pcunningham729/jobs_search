"""Tests for evaluation.py.

The CSV tests write small files into `tmp_path`. The end-to-end tests use a
temporary store and Chroma index with a fake embedder that counts keywords
(the same idea as in test_rag.py and test_matching.py).
"""

import pytest

from jobs_search.evaluation import (
    EvalReport,
    LabeledJob,
    ScoredLabel,
    append_history,
    evaluate,
    find_good_below_bad,
    last_precision,
    load_labels,
    pick_jobs_to_label,
    precision_at_k,
    write_label_sheet,
)
from jobs_search.models import Job
from jobs_search.profile import Filters, Interest, Profile
from jobs_search.rag import index_jobs, open_index
from jobs_search.store import connect, save_jobs


def make_job(
    job_id: str, title: str, description: str, location: str = "Remote (US)"
) -> Job:
    return Job(
        id=job_id,
        source="lever",
        company="acme",
        title=title,
        location=location,
        url="https://example.com",
        description=description,
    )


def scored(title: str, label: str, score: float = 0.0) -> ScoredLabel:
    return ScoredLabel(
        job_id=title, title=title, label=label, score=score, passes_filters=True
    )


# ---------- load_labels ----------


def test_load_labels_skips_empty_rows_and_ignores_case(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "job_id,label,note,title\n"
        "lever:acme:1,Good,nice,Data Engineer\n"
        "lever:acme:2,,,Not labeled yet\n"
        "lever:acme:3,BAD,,Sales\n",
        encoding="utf-8",
    )

    labels = load_labels(path)

    assert labels == [
        LabeledJob(job_id="lever:acme:1", label="good", note="nice"),
        LabeledJob(job_id="lever:acme:3", label="bad"),
    ]


def test_load_labels_handles_files_saved_by_excel(tmp_path):
    # Excel adds an invisible "byte order mark" to the start of the file.
    path = tmp_path / "labels.csv"
    path.write_text("job_id,label\nlever:acme:1,good\n", encoding="utf-8-sig")

    assert load_labels(path)[0].job_id == "lever:acme:1"


def test_load_labels_rejects_unknown_labels_with_the_line_number(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "job_id,label\nlever:acme:1,good\nlever:acme:2,maybe\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="line 3"):
        load_labels(path)


# ---------- write_label_sheet ----------


def test_label_sheet_round_trip(tmp_path):
    path = tmp_path / "labeled_jobs.csv"
    write_label_sheet(path, [make_job("lever:acme:1", "Data Engineer", "Text.")])

    # A fresh sheet has no labels yet, so nothing loads...
    assert load_labels(path) == []
    # ...and the job is in it, ready to label.
    assert "lever:acme:1,,,Data Engineer" in path.read_text(encoding="utf-8")


def test_label_sheet_never_overwrites(tmp_path):
    path = tmp_path / "labeled_jobs.csv"
    path.write_text("my labels", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_label_sheet(path, [])
    assert path.read_text(encoding="utf-8") == "my labels"


# ---------- precision and ordering ----------


def test_precision_at_k():
    ranked = [
        scored("a", "good"),
        scored("b", "bad"),
        scored("c", "good"),
        scored("d", "bad"),
    ]

    assert precision_at_k(ranked, k=2) == 0.5
    assert precision_at_k(ranked, k=1) == 1.0
    assert precision_at_k([], k=10) == 0.0


def test_precision_uses_fewer_than_k_when_there_are_fewer_jobs():
    assert precision_at_k([scored("a", "good"), scored("b", "bad")], k=10) == 0.5


def test_find_good_below_bad():
    ranked = [
        scored("g1", "good"),
        scored("b1", "bad"),
        scored("g2", "good"),
        scored("b2", "bad"),
        scored("g3", "good"),
    ]

    assert find_good_below_bad(ranked) == {"g2": 1, "g3": 2}


# ---------- history ----------


def test_history_remembers_the_last_run(tmp_path):
    path = tmp_path / "data" / "eval_history.csv"
    assert last_precision(path) is None

    def report(precision: float) -> EvalReport:
        return EvalReport(
            precision_at_10=precision, good_count=5, bad_count=5, ranked=[],
            good_below_bad={}, good_filtered_out=[], missing_ids=[],
        )  # fmt: skip

    append_history(path, report(0.6), "first")
    append_history(path, report(0.8), "second")

    assert last_precision(path) == 0.8


# ---------- end to end ----------

KEYWORDS = ["data", "pipeline", "sales"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([float(lowered.count(word)) for word in KEYWORDS] + [1.0])
    return vectors


PROFILE = Profile(
    titles=["data engineer"],
    interests=[Interest(name="pipelines", text="data pipeline")],
    filters=Filters(locations=["Minneapolis"]),
)


@pytest.fixture
def setup(tmp_path):
    conn = connect(tmp_path / "jobs.db")
    index = open_index(tmp_path / "chroma")
    jobs = [
        make_job("lever:acme:data", "Data Engineer", "Data pipeline data pipeline."),
        make_job("lever:acme:sales", "Sales Lead", "Sales sales."),
        make_job(
            "lever:acme:paris", "Data Engineer", "Data pipeline.", location="Paris"
        ),
        make_job("lever:acme:other", "Data Analyst", "Some data."),
    ]
    save_jobs(conn, jobs)
    index_jobs(index, jobs, fake_embed)
    yield conn, index
    conn.close()


def test_evaluate_ranks_labeled_jobs_and_reports_problems(setup):
    conn, index = setup
    labels = [
        LabeledJob(job_id="lever:acme:sales", label="bad"),
        LabeledJob(job_id="lever:acme:data", label="good"),
        LabeledJob(job_id="lever:acme:paris", label="good"),
        LabeledJob(job_id="lever:acme:typo", label="good"),
    ]

    report = evaluate(conn, index, PROFILE, fake_embed, labels)

    assert report.ranked[-1].job_id == "lever:acme:sales"  # the bad job ranks last
    assert report.precision_at_10 == pytest.approx(2 / 3)
    assert report.good_below_bad == {}
    assert report.good_filtered_out == ["Data Engineer"]  # the Paris job
    assert report.missing_ids == ["lever:acme:typo"]


def test_pick_jobs_to_label_mixes_top_and_random(setup):
    conn, index = setup

    picked = pick_jobs_to_label(
        conn, index, PROFILE, fake_embed, top_count=1, random_count=5
    )
    ids = {job.id for job in picked}

    # 1 top job + the 2 others that pass the filters (only 2 are available).
    assert len(picked) == 3
    assert "lever:acme:paris" not in ids  # fails the location filter
    assert len(ids) == 3  # no job picked twice

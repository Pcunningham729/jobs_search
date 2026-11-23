"""Tests for rag.py.

These use a real Chroma index in a temporary folder (Chroma runs on your
computer, no network), but a FAKE embedder instead of the real model.

The fake embedder counts a few keywords, so texts about the same things get
similar vectors, just like a real model, only much cruder:
    "data pipeline" -> [1, 1, 0, 0, 1]   (data, pipeline, sales, manager, +1)
"""

import pytest

from jobs_search.models import Job
from jobs_search.rag import (
    chunk_text,
    get_job_vectors,
    get_title_vector,
    index_jobs,
    open_index,
    rebuild_index,
    search_titles,
    split_long_paragraph,
)
from jobs_search.store import connect, save_jobs

KEYWORDS = ["data", "pipeline", "sales", "manager"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Count each keyword in each text. The extra 1.0 at the end keeps a text
    with no keywords from becoming all zeros, which has no direction."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vector = []
        for word in KEYWORDS:
            vector.append(float(lowered.count(word)))
        vector.append(1.0)
        vectors.append(vector)
    return vectors


def make_job(job_id: str, title: str, description: str) -> Job:
    return Job(
        id=job_id,
        source="lever",
        company="acme",
        title=title,
        location="Remote",
        url="https://example.com",
        description=description,
    )


@pytest.fixture
def index(tmp_path):
    """A fresh, empty Chroma index for each test."""
    return open_index(tmp_path / "chroma")


# ---------- chunk_text ----------


def test_short_text_is_one_chunk():
    assert chunk_text("Build pipelines.\nShip them.") == [
        "Build pipelines.\nShip them."
    ]


def test_paragraphs_are_grouped_up_to_the_limit():
    # The first two lines plus the line break between them are exactly 31
    # characters, so they fit together. The third line won't fit, so it
    # starts a new chunk.
    text = "Short line.\nAnother short line.\nA third line here."

    chunks = chunk_text(text, max_chars=31)

    assert chunks == ["Short line.\nAnother short line.", "A third line here."]


def test_no_chunk_is_longer_than_the_limit():
    text = "\n".join(["word " * 50] * 10)  # 10 paragraphs of 250 characters

    for chunk in chunk_text(text, max_chars=300):
        assert len(chunk) <= 300


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n\n  ") == []


def test_long_paragraph_is_split_between_words():
    pieces = split_long_paragraph("aaa bbb ccc ddd", max_chars=7)

    assert pieces == ["aaa bbb", "ccc ddd"]


# ---------- index_jobs and get_job_vectors ----------


def test_index_jobs_stores_one_vector_per_chunk(index):
    job = make_job("lever:acme:1", "Data Engineer", "Build data pipelines.")

    count = index_jobs(index, [job], fake_embed)

    assert count == 1
    vectors = get_job_vectors(index, "lever:acme:1")
    assert len(vectors) == 1
    # "data" once, "pipeline" once, no "sales", no "manager", then the extra 1.0.
    # Chroma stores numbers in a compact format that rounds them very slightly
    # (1.0 comes back as 0.99999994), so compare with `pytest.approx`.
    assert vectors[0] == pytest.approx([1.0, 1.0, 0.0, 0.0, 1.0])


def test_reindexing_a_job_replaces_its_old_chunks(index):
    long_description = "\n".join(["data " * 100] * 5)  # several chunks
    index_jobs(
        index, [make_job("lever:acme:1", "Data Engineer", long_description)], fake_embed
    )
    assert len(get_job_vectors(index, "lever:acme:1")) > 1

    index_jobs(
        index, [make_job("lever:acme:1", "Data Engineer", "Short now.")], fake_embed
    )

    assert len(get_job_vectors(index, "lever:acme:1")) == 1


def test_unknown_job_has_no_vectors(index):
    assert get_job_vectors(index, "lever:acme:nope") == []


# ---------- search_titles ----------


def test_search_titles_puts_the_closest_title_first(index):
    jobs = [
        make_job("lever:acme:1", "Sales Manager", "Sell things."),
        make_job("lever:acme:2", "Data Engineer, Pipelines", "Build data pipelines."),
    ]
    index_jobs(index, jobs, fake_embed)

    matches = search_titles(index, "data pipeline", fake_embed)

    assert matches[0].job_id == "lever:acme:2"
    assert matches[0].title == "Data Engineer, Pipelines"
    assert matches[0].company == "acme"
    assert matches[0].similarity > matches[1].similarity


def test_search_titles_on_an_empty_index_returns_nothing(index):
    assert search_titles(index, "data engineer", fake_embed) == []


# ---------- rebuild_index ----------


def test_rebuild_index_uses_the_store_as_the_source(index, tmp_path):
    conn = connect(tmp_path / "jobs.db")
    save_jobs(conn, [make_job("lever:acme:1", "Data Engineer", "Build pipelines.")])
    # Put a job in the index that is NOT in the store.
    index_jobs(index, [make_job("lever:acme:old", "Old Job", "Gone.")], fake_embed)

    rebuild_index(conn, index, fake_embed)

    # The store's job is indexed; the stray job is gone.
    assert len(get_job_vectors(index, "lever:acme:1")) == 1
    assert get_job_vectors(index, "lever:acme:old") == []
    conn.close()


# ---------- get_title_vector ----------


def test_get_title_vector_returns_the_stored_title_vector(index):
    index_jobs(index, [make_job("lever:acme:1", "Data Engineer", "Text.")], fake_embed)

    vector = get_title_vector(index, "lever:acme:1")

    # "Data Engineer": "data" once, nothing else, then the extra 1.0.
    assert vector == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0])
    assert get_title_vector(index, "lever:acme:nope") is None

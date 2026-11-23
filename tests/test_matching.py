"""Tests for matching.py.

The filter and scoring functions are pure (no database, no model), so most
tests just pass in jobs and tiny hand-made vectors. The last tests run the
whole of rank_jobs() with a temporary store, a temporary Chroma index and a
fake embedder (it counts keywords, like the one in test_rag.py).
"""

from datetime import UTC, datetime

import pytest

from jobs_search.matching import (
    LabeledVector,
    ProfileVectors,
    cosine_similarity,
    is_recent,
    is_us_remote,
    mentions_us,
    passes_location,
    passes_salary,
    rank_jobs,
    score_job,
)
from jobs_search.models import Job
from jobs_search.profile import AvoidItem, Filters, Interest, Profile, Weights
from jobs_search.rag import index_jobs, open_index
from jobs_search.store import connect, save_jobs, set_status


def make_job(job_id: str = "lever:acme:1", **changes) -> Job:
    """A test job. Pass field names to change any of them."""
    fields = {
        "id": job_id,
        "source": "lever",
        "company": "acme",
        "title": "Data Engineer",
        "location": "Minneapolis, MN",
        "url": "https://example.com",
        "description": "Build data pipelines.",
    }
    fields.update(changes)
    return Job(**fields)


TWIN_CITIES = Filters(locations=["Minneapolis", "St. Paul"], min_salary=100000)


# ---------- location ----------


@pytest.mark.parametrize(
    "location, expected",
    [
        ("Remote, United States", True),
        ("Remote (US)", True),
        ("Remote; New York, NY; San Francisco, CA", True),
        ("Remote", True),
        ("Remote, North America", True),
        ("Remote, Poland", False),
        ("Remote (Canada)", False),
        ("Remote; Berlin", False),
        ("Minneapolis, MN", False),  # in a good place, but not remote
    ],
)
def test_is_us_remote(location, expected):
    # `parametrize` runs this one test once for each (location, expected) pair.
    assert is_us_remote(location) == expected


def test_mentions_us_does_not_confuse_other_words():
    assert mentions_us("Austin, TX") is True
    assert mentions_us("London, UK") is False
    assert mentions_us("Russia") is False  # contains "us", but isn't the US


def test_job_in_one_of_your_places_passes():
    assert passes_location(make_job(location="St. Paul, MN"), TWIN_CITIES) is True


def test_place_matching_ignores_case():
    assert passes_location(make_job(location="MINNEAPOLIS"), TWIN_CITIES) is True


def test_job_elsewhere_fails():
    assert passes_location(make_job(location="Chicago, IL"), TWIN_CITIES) is False


def test_us_remote_job_passes_only_if_remote_is_allowed():
    job = make_job(location="Remote, United States")

    assert passes_location(job, TWIN_CITIES) is True
    no_remote = Filters(locations=["Minneapolis"], allow_remote=False)
    assert passes_location(job, no_remote) is False


# ---------- salary ----------


def test_salary_range_reaching_the_minimum_passes():
    # The top of the range ($120K) reaches $100K.
    job = make_job(salary_min=90000, salary_max=120000)

    assert passes_salary(job, TWIN_CITIES) is True


def test_salary_range_below_the_minimum_fails():
    job = make_job(salary_min=60000, salary_max=90000)

    assert passes_salary(job, TWIN_CITIES) is False


def test_only_a_minimum_salary_is_used_when_there_is_no_maximum():
    assert passes_salary(make_job(salary_min=110000), TWIN_CITIES) is True


def test_missing_salary_follows_your_setting():
    job = make_job()  # no salary listed

    assert passes_salary(job, Filters(min_salary=100000)) is True
    strict = Filters(min_salary=100000, keep_if_salary_missing=False)
    assert passes_salary(job, strict) is False


def test_no_minimum_means_any_salary_passes():
    assert passes_salary(make_job(salary_max=10), Filters()) is True


# ---------- is_recent ----------


def test_is_recent():
    now = datetime(2026, 11, 22, tzinfo=UTC)

    assert is_recent(make_job(posted_at="2026-11-20T00:00:00+00:00"), 7, now) is True
    assert is_recent(make_job(posted_at="2026-10-01T00:00:00+00:00"), 7, now) is False
    assert is_recent(make_job(), 7, now) is True  # no date: kept


# ---------- scoring ----------


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(
        1.0
    )  # same direction
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)  # unrelated
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # no direction


def test_score_job_uses_the_best_title_interest_and_avoid():
    profile_vectors = ProfileVectors(
        titles=[
            LabeledVector(label="far title", vector=[0.0, 1.0]),
            LabeledVector(label="exact title", vector=[1.0, 0.0]),
        ],
        interests=[
            LabeledVector(label="pipelines", vector=[1.0, 0.0]),
            LabeledVector(label="sales", vector=[0.0, 1.0]),
        ],
        avoid=[LabeledVector(label="travel", vector=[0.0, 1.0])],
    )
    # One chunk exactly like "pipelines", and nothing like "sales" or "travel".
    scores = score_job(
        title_vector=[1.0, 0.0],
        chunk_vectors=[[1.0, 0.0]],
        profile_vectors=profile_vectors,
        weights=Weights(title=1.0, interests=1.0, avoid=0.5),
    )

    assert scores.title == pytest.approx(1.0)  # the exact title wins
    assert scores.best_interest == "pipelines"
    assert scores.interest == pytest.approx(1.0)
    assert scores.interests["sales"] == pytest.approx(0.0)
    assert scores.avoid == pytest.approx(0.0)
    assert scores.total == pytest.approx(2.0)  # 1.0 * 1 + 1.0 * 1 - 0.5 * 0


def test_avoid_match_lowers_the_total():
    profile_vectors = ProfileVectors(
        titles=[LabeledVector(label="t", vector=[1.0, 0.0])],
        interests=[LabeledVector(label="i", vector=[1.0, 0.0])],
        avoid=[LabeledVector(label="travel", vector=[1.0, 0.0])],
    )

    scores = score_job([1.0, 0.0], [[1.0, 0.0]], profile_vectors, Weights(avoid=0.5))

    assert scores.avoid == pytest.approx(1.0)
    assert scores.total == pytest.approx(1.5)  # 1 + 1 - 0.5


# ---------- rank_jobs, end to end ----------

KEYWORDS = ["data", "pipeline", "sales", "travel"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Count each keyword in each text, plus a constant 1.0 (see test_rag.py)."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([float(lowered.count(word)) for word in KEYWORDS] + [1.0])
    return vectors


PROFILE = Profile(
    titles=["data engineer"],
    interests=[Interest(name="pipelines", text="data pipeline work")],
    avoid=[AvoidItem(text="travel")],
    filters=Filters(locations=["Minneapolis"], min_salary=100000),
)


@pytest.fixture
def setup(tmp_path):
    """A store and an index holding four jobs, ready for rank_jobs()."""
    conn = connect(tmp_path / "jobs.db")
    index = open_index(tmp_path / "chroma")
    jobs = [
        make_job("lever:acme:data", title="Data Engineer",
                 description="Build data pipeline tools. Data pipeline on-call."),
        make_job("lever:acme:sales", title="Sales Lead",
                 description="Sales sales sales."),
        make_job("lever:acme:travel", title="Data Engineer, Field",
                 description="Data pipeline work with travel travel travel."),
        make_job("lever:acme:chicago", title="Data Engineer",
                 location="Chicago, IL", description="Data pipeline work."),
    ]  # fmt: skip
    save_jobs(conn, jobs)
    index_jobs(index, jobs, fake_embed)
    yield conn, index
    conn.close()


def test_rank_jobs_filters_then_ranks_best_first(setup):
    conn, index = setup

    results = rank_jobs(conn, index, PROFILE, fake_embed)
    ids = [result.job.id for result in results]

    assert "lever:acme:chicago" not in ids  # filtered out: wrong location
    assert ids[0] == "lever:acme:data"  # best title + interest, no travel
    # The travel job matches the avoid list, so it ranks below the clean one.
    assert ids.index("lever:acme:travel") > ids.index("lever:acme:data")
    assert results[0].scores.best_interest == "pipelines"


def test_rank_jobs_skips_jobs_you_have_acted_on(setup):
    conn, index = setup
    set_status(conn, "lever:acme:data", "rejected")

    ids = [result.job.id for result in rank_jobs(conn, index, PROFILE, fake_embed)]

    assert "lever:acme:data" not in ids


def test_rank_jobs_respects_the_limit(setup):
    conn, index = setup

    assert len(rank_jobs(conn, index, PROFILE, fake_embed, limit=1)) == 1


def test_title_argument_replaces_profile_titles(setup):
    conn, index = setup

    results = rank_jobs(conn, index, PROFILE, fake_embed, title="sales")

    # Searching for "sales" moves the sales job's title score to the top.
    by_id = {result.job.id: result for result in results}
    assert (
        by_id["lever:acme:sales"].scores.title > by_id["lever:acme:data"].scores.title
    )

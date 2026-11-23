"""Rank saved jobs against your profile. No LLM: fast, free, and testable offline.

It works in two passes:

1. FILTERS: simple yes/no checks that drop jobs you'd never want.
   - Location: in one of your places, or remote in the US.
   - Salary: the top of the range reaches your minimum (or no salary is
     listed and you've said to keep those).

2. SCORING: for each job that passes, compare vectors (lists of numbers
   that represent meaning, see embeddings.py):
   - title score:    how close the job title is to your closest title
   - interest score: how close the job's best-matching chunk is to your
                     best-matching interest
   - avoid score:    how close the job comes to anything on your avoid list
   total = title weight * title + interests weight * interest
           - avoid weight * avoid

Job vectors were stored when the jobs were indexed (rag.py). Profile vectors
are calculated fresh on every call, so edits to profile.toml apply at once.
"""

import math
import re
import sqlite3
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from jobs_search.embeddings import Embedder
from jobs_search.models import Job
from jobs_search.profile import Filters, Profile, Weights
from jobs_search.rag import JobIndex, get_job_vectors, get_title_vector
from jobs_search.store import Status, list_jobs

# ---------------------------------------------------------------------------
# Pass 1: filters
# ---------------------------------------------------------------------------

# Words that show a remote job is open to people in the United States.
# ("North America" and "Americas" usually include the US.)
US_WORDS = ["united states", "usa", "u.s.", "north america", "americas"]

# Two-letter US state codes, to recognise places like "Austin, TX".
US_STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
]  # fmt: skip


def mentions_us(text: str) -> bool:
    """Does this location text point to the United States?

    Examples:
        "Remote, United States" -> True     "Remote (US)" -> True
        "New York, NY"          -> True     "Remote, Poland" -> False
        "London, UK"            -> False
    """
    lowered = text.lower()
    for word in US_WORDS:
        if word in lowered:
            return True

    # "US" as a whole word, in capitals (so the "us" inside "Russia" doesn't count).
    # In a regular expression, \b means "the edge of a word".
    if re.search(r"\bUS\b", text):
        return True

    # A state code after a comma, like ", TX".
    for code in US_STATE_CODES:
        if re.search(rf",\s*{code}\b", text):
            return True

    return False


def is_us_remote(location: str) -> bool:
    """Is this a remote job that people in the US can do?

    Locations can list several places separated by ";", e.g.
    "Remote; New York, NY; San Francisco, CA".

    Examples:
        "Remote, United States"          -> True
        "Remote; New York, NY"           -> True  (plain "Remote" + US places)
        "Remote"                         -> True  (no country given, keep it)
        "Remote, Poland"                 -> False
        "Remote; Berlin"                 -> False (plain "Remote" + non-US place)
        "Minneapolis, MN"                -> False (not remote)
    """
    places = []
    for place in location.split(";"):
        place = place.strip()
        if place:
            places.append(place)

    remote_places = []
    other_places = []
    for place in places:
        if "remote" in place.lower():
            remote_places.append(place)
        else:
            other_places.append(place)

    if not remote_places:
        return False  # not a remote job at all

    # A remote place that names the US: "Remote, United States", "Remote (US)".
    for place in remote_places:
        if mentions_us(place):
            return True

    # Only a plain "Remote" with no country. Decide from the other places:
    # none at all -> keep it; otherwise keep it only if one is in the US.
    for place in remote_places:
        if place.lower() == "remote":
            if not other_places:
                return True
            for other in other_places:
                if mentions_us(other):
                    return True

    return False


def passes_location(job: Job, filters: Filters) -> bool:
    """Is the job in one of your places, or remote in the US (if allowed)?"""
    location = job.location.lower()
    for place in filters.locations:
        if place.lower() in location:
            return True
    return filters.allow_remote and is_us_remote(job.location)


def passes_salary(job: Job, filters: Filters) -> bool:
    """Does the job's salary reach your minimum?

    Uses the TOP of the range: a $90K-$120K job can pay your $100K, so it passes.
    """
    if filters.min_salary is None:
        return True  # you didn't set a minimum

    # The top of the range, or the bottom if only that is given.
    top = job.salary_max
    if top is None:
        top = job.salary_min

    if top is None:
        return filters.keep_if_salary_missing  # no salary listed
    return top >= filters.min_salary


def passes_filters(job: Job, filters: Filters) -> bool:
    """Does the job pass every filter?"""
    return passes_location(job, filters) and passes_salary(job, filters)


def is_recent(job: Job, days: int, now: datetime | None = None) -> bool:
    """Was the job posted in the last `days` days? Jobs with no date are kept.

    `now` is only for tests, so they don't depend on today's date.
    """
    if job.posted_at is None:
        return True
    if now is None:
        now = datetime.now(UTC)

    posted = job.posted_at
    if posted.tzinfo is None:
        # A date with no time zone: assume UTC, so it can be compared with `now`.
        posted = posted.replace(tzinfo=UTC)
    return posted >= now - timedelta(days=days)


# ---------------------------------------------------------------------------
# Pass 2: scoring
# ---------------------------------------------------------------------------


class LabeledVector(BaseModel):
    """A vector plus the text it came from, e.g. an interest's name."""

    label: str
    vector: list[float]


class ProfileVectors(BaseModel):
    """Your profile, turned into vectors."""

    titles: list[LabeledVector]
    interests: list[LabeledVector]
    avoid: list[LabeledVector]


class Scores(BaseModel):
    """How one job scored, piece by piece, so the result can be explained."""

    title: float  # best similarity between the job title and one of your titles
    interest: float  # best similarity between any interest and any chunk
    best_interest: str  # the name of that best-matching interest
    interests: dict[str, float]  # every interest's best similarity, by name
    avoid: float  # closest any avoid item came to any chunk (0 if no avoid list)
    total: float  # the weighted combination used for ranking


class MatchResult(BaseModel):
    """One ranked job."""

    job: Job
    scores: Scores


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How alike two vectors are: 1.0 = same direction (same meaning), 0 = unrelated.

    It's the "dot product" (multiply matching numbers, add them all up)
    divided by both vectors' lengths, so only their direction matters.
    """
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        return 0.0  # an all-zero vector has no direction
    return dot / (length_a * length_b)


def best_similarity(vector: list[float], others: list[list[float]]) -> float:
    """The highest similarity between `vector` and any vector in `others`."""
    best = 0.0
    for other in others:
        best = max(best, cosine_similarity(vector, other))
    return best


def embed_labeled(
    texts: list[str], labels: list[str], embed: Embedder
) -> list[LabeledVector]:
    """Embed `texts` and pair each vector with its label."""
    if not texts:
        return []  # nothing to embed (e.g. an empty avoid list)
    vectors = embed(texts)
    labeled = []
    for label, vector in zip(labels, vectors):
        labeled.append(LabeledVector(label=label, vector=vector))
    return labeled


def embed_profile(
    profile: Profile, embed: Embedder, title: str | None = None
) -> ProfileVectors:
    """Turn the profile into vectors. Called fresh on every ranking.

    If `title` is given (e.g. "data engineer"), it's used instead of the
    profile's titles, for searches like `main.py match --title "..."`.
    """
    if title is not None:
        titles = [title]
    else:
        titles = profile.titles

    interest_texts = [interest.text for interest in profile.interests]
    interest_names = [interest.name for interest in profile.interests]
    avoid_texts = [item.text for item in profile.avoid]

    return ProfileVectors(
        titles=embed_labeled(titles, titles, embed),
        interests=embed_labeled(interest_texts, interest_names, embed),
        avoid=embed_labeled(avoid_texts, avoid_texts, embed),
    )


def score_job(
    title_vector: list[float],
    chunk_vectors: list[list[float]],
    profile_vectors: ProfileVectors,
    weights: Weights,
) -> Scores:
    """Score one job from its stored vectors. Pure maths: no database, no model."""
    # Title: the closest of your titles.
    title_score = 0.0
    for title in profile_vectors.titles:
        title_score = max(title_score, cosine_similarity(title.vector, title_vector))

    # Interests: each interest's best chunk, then the best interest overall.
    interest_scores = {}
    for interest in profile_vectors.interests:
        interest_scores[interest.label] = best_similarity(
            interest.vector, chunk_vectors
        )
    # `max(..., key=...)` picks the name whose score is highest.
    best_interest = max(interest_scores, key=interest_scores.get)
    interest_score = interest_scores[best_interest]

    # Avoid: the closest any avoid item comes to any chunk.
    avoid_score = 0.0
    for item in profile_vectors.avoid:
        avoid_score = max(avoid_score, best_similarity(item.vector, chunk_vectors))

    total = (
        weights.title * title_score
        + weights.interests * interest_score
        - weights.avoid * avoid_score
    )
    return Scores(
        title=title_score,
        interest=interest_score,
        best_interest=best_interest,
        interests=interest_scores,
        avoid=avoid_score,
        total=total,
    )


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def rank_jobs(
    conn: sqlite3.Connection,
    index: JobIndex,
    profile: Profile,
    embed: Embedder,
    title: str | None = None,
    new_since_days: int | None = None,
    statuses: list[Status] | None = None,
    limit: int = 20,
) -> list[MatchResult]:
    """Filter and score saved jobs, and return the best `limit` of them, best first.

    Arguments:
        title: search for this title instead of the profile's titles.
        new_since_days: only jobs posted in the last N days, e.g. 7.
        statuses: which statuses to include. Default: only "new", so jobs you
            applied to, rejected or hid don't come back.
        limit: how many results to return.
    """
    if statuses is None:
        statuses = ["new"]

    profile_vectors = embed_profile(profile, embed, title)

    results = []
    for job in list_jobs(conn, statuses=statuses):
        # Pass 1: cheap filters first, so we only score jobs that could fit.
        if not passes_filters(job, profile.filters):
            continue
        if new_since_days is not None and not is_recent(job, new_since_days):
            continue

        # Pass 2: score from the stored vectors.
        title_vector = get_title_vector(index, job.id)
        chunk_vectors = get_job_vectors(index, job.id)
        if title_vector is None or not chunk_vectors:
            continue  # not indexed yet, so there's nothing to compare

        scores = score_job(
            title_vector, chunk_vectors, profile_vectors, profile.weights
        )
        results.append(MatchResult(job=job, scores=scores))

    # Sort by total score, highest first. `key=` tells sort what to compare:
    # here a small unnamed function ("lambda") that takes a result and
    # returns its total score.
    results.sort(key=lambda result: result.scores.total, reverse=True)
    return results[:limit]

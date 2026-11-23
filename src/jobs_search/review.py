"""Have Claude rerank the shortlist, using your profile and resumes.

matching.py is fast but shallow: it compares embeddings (lists of numbers
that capture meaning), so it can't tell "Senior Data Scientist, 10+ years"
from "Data Science Intern". Claude can actually READ each posting next to
your resumes and judge the fit, like a friend who knows your background.

How it works:
    1. matching.rank_jobs() gives a shortlist (about 20 jobs).
    2. build_prompt() writes one message holding your profile, your resumes
       and every shortlisted job.
    3. Claude answers with STRUCTURED OUTPUT: instead of free text, the API
       makes the answer match a Pydantic model (ReviewBatch below), so we
       get Python objects back, never text we'd have to pick apart.
    4. review_jobs() matches each review to its job and sorts by fit.

This is one of only two places that calls an LLM (CLAUDE.md), so the call is
passed in as an argument (`ask`). Tests pass a fake `ask` and stay offline,
the same trick embeddings use with `embed`.

Try it (needs ANTHROPIC_API_KEY in .env, and costs a few cents):
    uv run jobs-search review
"""

import os
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from jobs_search.matching import MatchResult
from jobs_search.models import Job
from jobs_search.profile import Profile, Resume
from jobs_search.prompts import load_skill

DEFAULT_MODEL = "claude-opus-5-5"


def review_instructions() -> str:
    """The instructions Claude gets on every review.

    HOW to judge a posting comes from the screen-posting skill in
    agent_prompts/skills.md, so the chat agent and this review step judge
    jobs the same way, and you edit that in one place. Only the parts
    specific to reviewing a whole batch live here.
    """
    return (
        "You are helping one person decide which job postings are worth their "
        "time. You get their profile, their resumes and a shortlist of postings.\n\n"
        "Apply this skill to every posting:\n\n"
        + load_skill("screen-posting")
        + "\n\nReview every job_id you are given exactly once, and copy each "
        "job_id exactly as written."
    )


# ---------------------------------------------------------------------------
# What Claude sends back (the structured output)
# ---------------------------------------------------------------------------
# The `description=` text inside Field() is sent to Claude as part of the
# schema, so it doubles as instructions for that field.


class JobReview(BaseModel):
    """Claude's judgment of one job."""

    job_id: str = Field(description="The job_id exactly as given.")
    fit: int = Field(description="1 (clearly wrong) to 10 (apply today).")
    verdict: Literal["strong", "maybe", "skip"]
    reasons: list[str] = Field(description="Why it fits. Short and specific.")
    concerns: list[str] = Field(description="Why it might not. Can be empty.")


class ReviewBatch(BaseModel):
    """The whole answer: one review per shortlisted job."""

    reviews: list[JobReview]


class ReviewedJob(BaseModel):
    """A shortlisted job with both scores: the embedding one and Claude's."""

    job: Job
    match_score: float  # from matching.py
    review: JobReview | None  # None if Claude skipped this job


# A function that takes (instructions, message) and returns Claude's answer.
# call_claude() below is the real one; tests use a fake.
Asker = Callable[[str, str], ReviewBatch]


# ---------------------------------------------------------------------------
# Building the message
# ---------------------------------------------------------------------------

# Emails like "name@example.com" and US-style phone numbers like
# "(612) 555-0100" or "612.555.0100". `re` is Python's regular-expression
# module: a tiny language for describing text patterns.
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")


def remove_contact_details(text: str) -> str:
    """Blank out emails and phone numbers. Claude doesn't need them to judge fit.

    >>> remove_contact_details("Call 612-555-0100 or mail a@b.com")
    'Call [phone removed] or mail [email removed]'
    """
    text = EMAIL_PATTERN.sub("[email removed]", text)
    return PHONE_PATTERN.sub("[phone removed]", text)


def format_profile(profile: Profile) -> str:
    """The profile as readable text (Claude reads text, not Python objects)."""
    lines = ["Target titles: " + ", ".join(profile.titles), "", "Interests:"]
    for interest in profile.interests:
        lines.append(f"- {interest.name}: {interest.text}")
    if profile.avoid:
        lines.append("")
        lines.append("Avoid:")
        for item in profile.avoid:
            lines.append(f"- {item.text}")
    return "\n".join(lines)


def format_job(job: Job) -> str:
    """One job as text, with the job_id Claude must copy into its review."""
    lines = [
        f"job_id: {job.id}",
        f"Title: {job.title}",
        f"Company: {job.company}",
        f"Location: {job.location}",
    ]
    if job.salary_min or job.salary_max:
        lines.append(f"Salary: {job.salary_min} to {job.salary_max} USD per year")
    lines.append("")
    lines.append(job.description)
    return "\n".join(lines)


def build_prompt(profile: Profile, resumes: list[Resume], jobs: list[Job]) -> str:
    """The one message Claude gets: profile, resumes, then every job.

    Each part is wrapped in XML-style tags like <resume>...</resume>. Claude
    is trained to treat tags as clear boundaries, so it never confuses where
    one resume ends and a job begins.
    """
    parts = ["<profile>", format_profile(profile), "</profile>", ""]
    for resume in resumes:
        parts.append(f'<resume name="{resume.name}">')
        parts.append(remove_contact_details(resume.text))
        parts.append("</resume>")
        parts.append("")
    for job in jobs:
        parts.append("<job>")
        parts.append(format_job(job))
        parts.append("</job>")
        parts.append("")
    parts.append(f"Review all {len(jobs)} jobs above.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Calling Claude
# ---------------------------------------------------------------------------


def review_model_name() -> str:
    """Which Claude model reviews jobs. Override with JOBS_SEARCH_REVIEW_MODEL."""
    return os.environ.get("JOBS_SEARCH_REVIEW_MODEL", DEFAULT_MODEL)


def call_claude(instructions: str, message: str) -> ReviewBatch:
    """Send one review request to Claude and return its structured answer.

    The imports are inside the function, so tests (which never call this)
    don't need an API key or the network.
    """
    import anthropic
    from dotenv import load_dotenv

    # Copies ANTHROPIC_API_KEY from your .env file into the environment,
    # where the anthropic library looks for it. We never read the key here.
    load_dotenv()
    client = anthropic.Anthropic()

    # .parse() is the structured-output call: `output_format=ReviewBatch`
    # makes the answer fit our model, and the library checks it for us.
    #
    # Fallbacks: rarely, a safety check declines a request. With these two
    # settings the API retries on a fallback model inside the same call,
    # instead of just stopping. ("beta" means the feature is new.)
    response = client.beta.messages.parse(
        model=review_model_name(),
        max_tokens=16000,
        system=instructions,
        messages=[{"role": "user", "content": message}],
        output_format=ReviewBatch,
        # effort = how hard Claude thinks before answering (low to max).
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to review these jobs.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off. Try a shorter shortlist.")
    if response.parsed_output is None:
        raise RuntimeError("Claude's answer didn't match the ReviewBatch format.")
    return response.parsed_output


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def review_jobs(
    results: list[MatchResult],
    profile: Profile,
    resumes: list[Resume],
    ask: Asker = call_claude,
) -> list[ReviewedJob]:
    """Have Claude review the shortlist, then sort it best fit first.

    Jobs Claude skipped stay in the list (review=None) at the end, so you
    never silently lose one. Reviews for job_ids we didn't send are ignored.
    """
    if not results:
        return []

    jobs = [result.job for result in results]
    batch = ask(review_instructions(), build_prompt(profile, resumes, jobs))

    # A dict lets us look up each job's review by id. If Claude reviewed a
    # job twice, the later review wins.
    reviews_by_id = {}
    for review in batch.reviews:
        reviews_by_id[review.job_id] = review

    reviewed = []
    for result in results:
        reviewed.append(
            ReviewedJob(
                job=result.job,
                match_score=result.scores.total,
                review=reviews_by_id.get(result.job.id),  # None if missing
            )
        )

    reviewed.sort(key=sort_key, reverse=True)
    return reviewed


def sort_key(item: ReviewedJob) -> tuple[int, float]:
    """Sort by Claude's fit first, then by the match score to break ties.

    Python compares tuples one item at a time, so (8, 0.9) beats (8, 0.7)
    and both beat (7, 1.5). Unreviewed jobs get fit 0, so they sort last.
    """
    fit = item.review.fit if item.review else 0
    return (fit, item.match_score)

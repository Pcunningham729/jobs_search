"""Tests for review.py.

No test talks to Claude. review_jobs() takes the Claude call as an argument
(`ask`), so these tests pass a fake that returns a ready-made answer, and
record what it was sent.
"""

from jobs_search.matching import MatchResult, Scores
from jobs_search.models import Job
from jobs_search.profile import AvoidItem, Interest, Profile, Resume
from jobs_search.review import (
    JobReview,
    ReviewBatch,
    build_prompt,
    remove_contact_details,
    review_instructions,
    review_jobs,
)


def make_job(job_id: str, title: str) -> Job:
    return Job(
        id=job_id,
        source="lever",
        company="acme",
        title=title,
        location="Remote (US)",
        url="https://example.com",
        description=f"{title} description.",
    )


def make_result(job_id: str, title: str, total: float) -> MatchResult:
    scores = Scores(
        title=0.0, interest=0.0, best_interest="", interests={}, avoid=0.0,
        total=total,
    )  # fmt: skip
    return MatchResult(job=make_job(job_id, title), scores=scores)


def make_review(job_id: str, fit: int) -> JobReview:
    return JobReview(
        job_id=job_id, fit=fit, verdict="maybe", reasons=["ok"], concerns=[]
    )


PROFILE = Profile(
    titles=["data engineer"],
    interests=[Interest(name="pipelines", text="Building data pipelines.")],
    avoid=[AvoidItem(text="Lots of travel.")],
)
RESUMES = [Resume(name="main", text="Alex Example\nalex@example.com\n612-555-0100")]


class FakeClaude:
    """Stands in for call_claude(): returns `answer` and remembers its inputs."""

    def __init__(self, answer: ReviewBatch):
        self.answer = answer
        self.calls = []  # (instructions, message) for each call

    def __call__(self, instructions: str, message: str) -> ReviewBatch:
        self.calls.append((instructions, message))
        return self.answer


# ---------- building the prompt ----------


def test_contact_details_are_removed():
    text = remove_contact_details("Mail alex@example.com or call (612) 555-0100.")

    assert text == "Mail [email removed] or call [phone removed]."


def test_prompt_holds_profile_resume_and_every_job_but_no_contact_details():
    jobs = [
        make_job("lever:acme:1", "Data Engineer"),
        make_job("lever:acme:2", "Analyst"),
    ]

    prompt = build_prompt(PROFILE, RESUMES, jobs)

    assert "Target titles: data engineer" in prompt
    assert "- pipelines: Building data pipelines." in prompt
    assert "Lots of travel." in prompt
    assert "Alex Example" in prompt
    assert "job_id: lever:acme:1" in prompt
    assert "job_id: lever:acme:2" in prompt
    assert "alex@example.com" not in prompt
    assert "555-0100" not in prompt


# ---------- review_jobs ----------


def test_review_sorts_by_claude_fit_not_match_score():
    results = [
        make_result("lever:acme:1", "High match, poor fit", total=1.5),
        make_result("lever:acme:2", "Lower match, great fit", total=1.0),
    ]
    fake = FakeClaude(
        ReviewBatch(
            reviews=[make_review("lever:acme:1", 3), make_review("lever:acme:2", 9)]
        )
    )

    reviewed = review_jobs(results, PROFILE, RESUMES, ask=fake)

    assert [item.job.id for item in reviewed] == ["lever:acme:2", "lever:acme:1"]
    assert reviewed[0].match_score == 1.0
    assert fake.calls[0][0] == review_instructions()


def test_match_score_breaks_ties():
    results = [
        make_result("lever:acme:1", "A", total=0.5),
        make_result("lever:acme:2", "B", total=0.9),
    ]
    fake = FakeClaude(
        ReviewBatch(
            reviews=[make_review("lever:acme:1", 7), make_review("lever:acme:2", 7)]
        )
    )

    reviewed = review_jobs(results, PROFILE, RESUMES, ask=fake)

    assert reviewed[0].job.id == "lever:acme:2"


def test_jobs_claude_skipped_are_kept_last_and_unknown_ids_ignored():
    results = [
        make_result("lever:acme:1", "Skipped", total=2.0),
        make_result("lever:acme:2", "Reviewed", total=1.0),
    ]
    fake = FakeClaude(
        ReviewBatch(
            reviews=[make_review("lever:acme:2", 5), make_review("made-up-id", 10)]
        )
    )

    reviewed = review_jobs(results, PROFILE, RESUMES, ask=fake)

    assert len(reviewed) == 2  # nothing lost, nothing invented
    assert reviewed[-1].job.id == "lever:acme:1"
    assert reviewed[-1].review is None


def test_empty_shortlist_never_calls_claude():
    fake = FakeClaude(ReviewBatch(reviews=[]))

    assert review_jobs([], PROFILE, RESUMES, ask=fake) == []
    assert fake.calls == []


def test_instructions_use_the_screen_posting_skill():
    instructions = review_instructions()

    assert "## screen-posting" in instructions
    assert "exactly once" in instructions

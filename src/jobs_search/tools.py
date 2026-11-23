"""The job agent's tools: thin LangChain wrappers around our own modules.

A TOOL is a Python function the agent (Claude) may decide to call. LangChain
sends Claude each tool's name, its arguments (from the type hints) and its
docstring. Claude never sees the code, so the DOCSTRING IS ITS ONLY GUIDE:
it says when to use the tool, what each argument means, and what comes back.

Per CLAUDE.md, tools stay thin. The real work lives in matching.py and
store.py; these functions only adapt arguments and turn results into text
Claude can read.

Tools never take identity (whose profile, which database) as an argument.
That comes from a ToolContext, fixed when the agent starts, so Claude can't
pick a different person's data by passing a different argument.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool

from jobs_search import store
from jobs_search.embeddings import Embedder
from jobs_search.matching import MatchResult, rank_jobs
from jobs_search.models import Job
from jobs_search.profile import Profile
from jobs_search.rag import JobIndex
from jobs_search.store import connect, get_job, get_status


# @dataclass writes the boring __init__ for us: ToolContext(db_path=..., ...)
# just stores each value as an attribute (ctx.db_path, ctx.index, ...).
@dataclass
class ToolContext:
    """Everything the tools need, set up once when the agent starts."""

    db_path: Path
    index: JobIndex
    profile: Profile
    embed: Embedder

    def open_db(self) -> sqlite3.Connection:
        """A fresh database connection for one tool call.

        LangChain may run a tool on a different THREAD (a separate line of
        work inside the program), and a SQLite connection only works on the
        thread that opened it. Opening one per call avoids that problem.
        """
        return connect(self.db_path)


# ---------------------------------------------------------------------------
# Formatting results as text for Claude
# ---------------------------------------------------------------------------


def format_salary(job: Job) -> str:
    if job.salary_min and job.salary_max:
        return f"${job.salary_min:,} to ${job.salary_max:,}"  # :, adds commas
    if job.salary_min or job.salary_max:
        return f"${(job.salary_min or job.salary_max):,}"
    return "not listed"


def format_match(position: int, result: MatchResult) -> str:
    """One line per match, with what the agent needs to explain and cite it."""
    job = result.job
    scores = result.scores
    return (
        f"{position}. {job.title} at {job.company} ({job.location})\n"
        f"   fit score {scores.total:.2f} (title {scores.title:.2f}, "
        f"best interest '{scores.best_interest}' {scores.interest:.2f}, "
        f"avoid {scores.avoid:.2f}), salary {format_salary(job)}\n"
        f"   id: {job.id}\n"
        f"   url: {job.url}"
    )


def format_job_details(job: Job, status: str | None) -> str:
    posted = job.posted_at.date().isoformat() if job.posted_at else "unknown"
    return (
        f"{job.title} at {job.company}\n"
        f"id: {job.id}\n"
        f"url: {job.url}\n"
        f"location: {job.location}\n"
        f"salary: {format_salary(job)}\n"
        f"posted: {posted}\n"
        f"your status: {status}\n\n"
        f"{job.description}"
    )


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def make_tools(ctx: ToolContext) -> list[BaseTool]:
    """Build the agent's tools, all sharing one ToolContext.

    The functions are defined INSIDE this one, so each can use `ctx` without
    taking it as an argument. (A function that remembers variables from
    around it like this is called a CLOSURE.) @tool turns each into a
    LangChain tool, using its name, type hints and docstring.
    """

    @tool
    def find_matches(
        title: str | None = None,
        new_since_days: int | None = None,
        limit: int = 10,
    ) -> str:
        """Find saved job postings that best fit the person, best first.

        Use this whenever the person wants job suggestions. Results already
        pass their location and salary filters, and skip jobs they applied
        to, rejected or hid.

        Args:
            title: Search for this job title instead of the titles in their
                profile, e.g. "data engineer". Leave empty to use their profile.
            new_since_days: Only jobs posted in the last this-many days, e.g. 7
                for "this week". Leave empty for any date.
            limit: How many jobs to return, e.g. 5. At most 25.

        Returns:
            A numbered list. Each job has its title, company, location, a fit
            score (higher is better; about 1.0 to 1.5 is typical, and the parts
            show what drove it), salary, id and url.
        """
        conn = ctx.open_db()
        try:
            results = rank_jobs(
                conn,
                ctx.index,
                ctx.profile,
                ctx.embed,
                title=title,
                new_since_days=new_since_days,
                limit=min(limit, 25),
            )
        finally:
            conn.close()  # `finally` runs even if rank_jobs raised an error

        if not results:
            return (
                "No saved jobs matched. Try a different title or a longer date range."
            )
        lines = []
        for position, result in enumerate(results, start=1):
            lines.append(format_match(position, result))
        return "\n\n".join(lines)

    @tool
    def get_job_details(job_id: str) -> str:
        """Get one job's full posting, to explain or judge it.

        Use this before saying anything specific about a job's duties or
        requirements, e.g. when the person asks "why this one?" or "is this
        right for me?".

        Args:
            job_id: The job's id exactly as find_matches showed it, e.g.
                "lever:spotify:1a2b3c".

        Returns:
            The title, company, id, url, location, salary, posted date, the
            person's status for it (new, applied, rejected or hidden), then
            the full description.
        """
        conn = ctx.open_db()
        try:
            job = get_job(conn, job_id)
            status = get_status(conn, job_id)
        finally:
            conn.close()

        if job is None:
            return f"No job with id {job_id!r}. Use an id from find_matches."
        return format_job_details(job, status)

    @tool
    def set_status(
        job_id: str, status: Literal["applied", "rejected", "hidden", "new"]
    ) -> str:
        """Record what the person did with a job.

        Use this when they say they applied to a job, aren't interested, or
        want it out of their results. Only call it when it's clear which job
        they mean.

        Args:
            job_id: The job's id exactly as find_matches showed it.
            status: "applied" (they applied), "rejected" (not interested),
                "hidden" (just stop showing it), or "new" (undo, show it again).
                Anything but "new" removes it from future find_matches results.

        Returns:
            A one-line confirmation, or an error message if the id is unknown.
        """
        conn = ctx.open_db()
        try:
            store.set_status(conn, job_id, status)  # the real work, in store.py
            job = get_job(conn, job_id)
        except ValueError as error:
            # Hand the error back as text, so the agent can tell the person
            # (or fix the id) instead of the whole chat crashing.
            return f"Couldn't update: {error}"
        finally:
            conn.close()

        return f"Marked {job.title} at {job.company} ({job_id}) as {status}."

    return [find_matches, get_job_details, set_status]

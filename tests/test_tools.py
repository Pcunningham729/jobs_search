"""Tests for tools.py.

Each tool is called with .invoke({...}), exactly how the agent calls it,
against a temporary store and index with a fake keyword-counting embedder.
"""

import pytest

from jobs_search.models import Job
from jobs_search.profile import Filters, Interest, Profile
from jobs_search.rag import index_jobs, open_index
from jobs_search.store import connect, get_status, save_jobs
from jobs_search.tools import ToolContext, make_tools

KEYWORDS = ["data", "pipeline", "sales"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([float(lowered.count(word)) for word in KEYWORDS] + [1.0])
    return vectors


def make_job(job_id: str, title: str, description: str) -> Job:
    return Job(
        id=job_id,
        source="lever",
        company="acme",
        title=title,
        location="Minneapolis, MN",
        url=f"https://example.com/{job_id}",
        description=description,
        salary_min=110000,
        salary_max=140000,
    )


PROFILE = Profile(
    titles=["data engineer"],
    interests=[Interest(name="pipelines", text="data pipeline")],
    filters=Filters(locations=["Minneapolis"]),
)


@pytest.fixture
def tools(tmp_path):
    """The three tools as a dict by name, over a store holding two jobs."""
    db_path = tmp_path / "jobs.db"
    index = open_index(tmp_path / "chroma")
    jobs = [
        make_job("lever:acme:data", "Data Engineer", "Data pipeline data pipeline."),
        make_job("lever:acme:sales", "Sales Lead", "Sales sales."),
    ]
    conn = connect(db_path)
    save_jobs(conn, jobs)
    conn.close()
    index_jobs(index, jobs, fake_embed)

    ctx = ToolContext(db_path=db_path, index=index, profile=PROFILE, embed=fake_embed)
    tool_list = make_tools(ctx)
    by_name = {}
    for each in tool_list:
        by_name[each.name] = each
    by_name["db_path"] = db_path  # handy for checking the database afterwards
    return by_name


def test_tool_names_match_what_agent_prompts_mention(tools):
    # agent_prompts/ tells the agent to call tools by these names.
    assert {"find_matches", "get_job_details", "set_status"} <= set(tools)


def test_find_matches_lists_best_first_with_id_and_url(tools):
    text = tools["find_matches"].invoke({})

    assert text.startswith("1. Data Engineer at acme")
    assert "id: lever:acme:data" in text
    assert "url: https://example.com/lever:acme:data" in text
    assert "$110,000 to $140,000" in text
    assert text.index("Data Engineer") < text.index("Sales Lead")


def test_find_matches_respects_limit(tools):
    text = tools["find_matches"].invoke({"limit": 1})

    assert "Sales Lead" not in text


def test_find_matches_says_so_when_nothing_matches(tools):
    # Hide both jobs, so nothing is left to suggest.
    tools["set_status"].invoke({"job_id": "lever:acme:data", "status": "hidden"})
    tools["set_status"].invoke({"job_id": "lever:acme:sales", "status": "hidden"})

    assert tools["find_matches"].invoke({}).startswith("No saved jobs matched")


def test_get_job_details(tools):
    text = tools["get_job_details"].invoke({"job_id": "lever:acme:data"})

    assert "Data Engineer at acme" in text
    assert "your status: new" in text
    assert "Data pipeline data pipeline." in text


def test_get_job_details_unknown_id(tools):
    text = tools["get_job_details"].invoke({"job_id": "nope"})

    assert text.startswith("No job with id 'nope'")


def test_set_status_saves_and_confirms(tools):
    text = tools["set_status"].invoke(
        {"job_id": "lever:acme:data", "status": "applied"}
    )

    assert text == "Marked Data Engineer at acme (lever:acme:data) as applied."
    conn = connect(tools["db_path"])
    assert get_status(conn, "lever:acme:data") == "applied"
    conn.close()


def test_set_status_unknown_id_returns_an_error_message(tools):
    text = tools["set_status"].invoke({"job_id": "nope", "status": "hidden"})

    assert text.startswith("Couldn't update:")

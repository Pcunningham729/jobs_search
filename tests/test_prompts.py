"""Tests for prompts.py.

Most tests write tiny prompt files into `tmp_path`. The last ones read the
real agent_prompts/ folder, so a typo there (like a broken "## " heading)
gets caught by the tests too.
"""

from pathlib import Path

import pytest

from jobs_search.prompts import (
    agent_system_prompt,
    load_prompt,
    load_skill,
    prompts_dir,
    split_skills,
)

SKILLS_TEXT = """# Skills

## find-new-jobs

**Use when:** they ask what's new.

1. Call find_matches.

## track-applications

1. Call set_status.
"""


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "system.md").write_text("\nYou help one person.\n\n", encoding="utf-8")
    (tmp_path / "skills.md").write_text(SKILLS_TEXT, encoding="utf-8")
    return tmp_path


def test_load_prompt_strips_blank_edges(folder):
    assert load_prompt("system", folder) == "You help one person."


def test_missing_prompt_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.md"):
        load_prompt("nope", tmp_path)


def test_split_skills_finds_each_section():
    skills = split_skills(SKILLS_TEXT)

    assert list(skills) == ["find-new-jobs", "track-applications"]
    assert skills["find-new-jobs"].startswith("## find-new-jobs")
    assert "1. Call find_matches." in skills["find-new-jobs"]
    assert "set_status" not in skills["find-new-jobs"]  # stops at the next skill


def test_load_skill_and_unknown_skill(folder):
    assert load_skill("track-applications", folder) == (
        "## track-applications\n\n1. Call set_status."
    )
    with pytest.raises(ValueError, match="find-new-jobs, track-applications"):
        load_skill("nope", folder)


def test_agent_system_prompt_is_system_then_skills(folder):
    prompt = agent_system_prompt(folder)

    assert prompt.startswith("You help one person.\n\n# Skills")


def test_prompts_dir_setting(monkeypatch):
    monkeypatch.delenv("JOBS_SEARCH_PROMPTS_DIR", raising=False)
    assert prompts_dir() == Path("agent_prompts")

    monkeypatch.setenv("JOBS_SEARCH_PROMPTS_DIR", "elsewhere")
    assert prompts_dir() == Path("elsewhere")


# ---------- the real agent_prompts/ folder ----------

# Path(__file__) is this test file; .parent.parent is the repo root.
REAL_FOLDER = Path(__file__).parent.parent / "agent_prompts"


def test_real_skills_file_has_the_expected_skills():
    skills = split_skills(load_prompt("skills", REAL_FOLDER))

    for name in [
        "find-new-jobs",
        "explain-a-match",
        "screen-posting",
        "track-applications",
    ]:
        assert name in skills


def test_real_system_prompt_loads():
    assert "find_matches" in agent_system_prompt(REAL_FOLDER)

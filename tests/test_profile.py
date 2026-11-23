"""Tests for profile.py.

Most tests read the made-up profile in tests/fixtures/profile/. The ones
that need a broken profile write a small profile.toml into `tmp_path`, a
temporary folder pytest makes for each test.
"""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from jobs_search.profile import load_profile, load_resumes, profile_dir

FIXTURE_DIR = Path("tests/fixtures/profile")

# The smallest valid profile: one title and one interest.
MINIMAL_TOML = """
titles = ["data engineer"]

[[interests]]
name = "pipelines"
text = "Building data pipelines."
"""


def write_profile(folder: Path, toml_text: str) -> Path:
    """Write `toml_text` as folder/profile.toml and return the folder."""
    (folder / "profile.toml").write_text(toml_text, encoding="utf-8")
    return folder


# ---------- load_profile ----------


def test_fixture_profile_loads():
    # This also keeps the format example in tests/fixtures/ valid.
    profile = load_profile(FIXTURE_DIR)

    assert profile.titles == ["data engineer", "analytics engineer"]
    assert profile.interests[0].name == "pipelines"
    assert len(profile.interests) == 2
    assert profile.avoid[0].text.startswith("Roles with heavy on-call")
    assert profile.filters.min_salary == 100000
    assert "Minneapolis" in profile.filters.locations


def test_filters_and_avoid_are_optional(tmp_path):
    profile = load_profile(write_profile(tmp_path, MINIMAL_TOML))

    assert profile.avoid == []
    assert profile.filters.allow_remote is True
    assert profile.filters.min_salary is None
    assert profile.filters.keep_if_salary_missing is True


def test_misspelled_field_is_rejected(tmp_path):
    bad = MINIMAL_TOML.replace("titles", "tittles")

    with pytest.raises(ValidationError):
        load_profile(write_profile(tmp_path, bad))


def test_misspelled_filter_is_rejected(tmp_path):
    bad = MINIMAL_TOML + "\n[filters]\nmin_sallary = 100000\n"

    with pytest.raises(ValidationError):
        load_profile(write_profile(tmp_path, bad))


def test_profile_needs_at_least_one_interest(tmp_path):
    with pytest.raises(ValidationError):
        load_profile(write_profile(tmp_path, 'titles = ["data engineer"]\n'))


def test_salary_must_be_a_number(tmp_path):
    bad = MINIMAL_TOML + '\n[filters]\nmin_salary = "lots"\n'

    with pytest.raises(ValidationError):
        load_profile(write_profile(tmp_path, bad))


def test_toml_syntax_error_is_reported(tmp_path):
    with pytest.raises(tomllib.TOMLDecodeError):
        load_profile(write_profile(tmp_path, 'titles = ["missing the closing quote]\n'))


def test_missing_profile_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path)


# ---------- profile_dir ----------


def test_profile_dir_defaults_to_profile(monkeypatch):
    monkeypatch.delenv("JOBS_SEARCH_PROFILE_DIR", raising=False)

    assert profile_dir() == Path("profile")


def test_profile_dir_comes_from_the_setting(monkeypatch):
    monkeypatch.setenv("JOBS_SEARCH_PROFILE_DIR", "tests/fixtures/profile")

    assert profile_dir() == Path("tests/fixtures/profile")


# ---------- load_resumes ----------


def test_fixture_resume_loads():
    resumes = load_resumes(FIXTURE_DIR)

    assert len(resumes) == 1
    assert resumes[0].name == "example-data-engineer"
    assert resumes[0].text.startswith("# Alex Example")


def test_resumes_are_sorted_and_hidden_files_skipped(tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    (folder / "b-analyst.txt").write_text("Analyst resume", encoding="utf-8")
    (folder / "a-engineer.md").write_text("Engineer resume", encoding="utf-8")
    (folder / ".gitkeep").write_text("", encoding="utf-8")

    resumes = load_resumes(tmp_path)

    assert [resume.name for resume in resumes] == ["a-engineer", "b-analyst"]


def test_unsupported_resume_type_is_an_error(tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    (folder / "resume.pdf").write_bytes(b"%PDF-1.7")

    with pytest.raises(ValueError, match="resume.pdf"):
        load_resumes(tmp_path)


def test_no_resumes_folder_means_no_resumes(tmp_path):
    assert load_resumes(tmp_path) == []


def test_weights_have_defaults_and_can_be_set(tmp_path):
    default = load_profile(write_profile(tmp_path, MINIMAL_TOML))
    assert default.weights.title == 1.0
    assert default.weights.avoid == 0.5

    custom = load_profile(
        write_profile(tmp_path, MINIMAL_TOML + "\n[weights]\navoid = 2.0\n")
    )
    assert custom.weights.avoid == 2.0
    assert custom.weights.interests == 1.0  # not given, so the default

"""Load your profile: what you're looking for, and your resumes.

The profile folder (default `profile/`, gitignored) looks like this:

    profile/
      profile.toml     <- titles, interests, avoid list, filters
      resumes/
        data-engineer.md
        analytics.txt

`load_profile()` reads profile.toml and checks it with the Pydantic classes
below, so a missing field or a typo gives a clear error right away.
`load_resumes()` reads the resume files, for review.py to show Claude.

The folder's location comes from the JOBS_SEARCH_PROFILE_DIR setting.
tests/fixtures/profile/ is a made-up profile in the same format.
"""

import os
import tomllib  # built into Python: reads TOML files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Resume file types we can read. PDF and Word files need extra libraries and
# their text often comes out jumbled, so convert those to .md or .txt.
RESUME_TYPES = [".md", ".txt"]


class StrictModel(BaseModel):
    """A BaseModel that refuses fields it doesn't know.

    Normally Pydantic silently ignores unknown fields. With extra="forbid",
    a typo in profile.toml like `intrests` becomes an error that names it,
    instead of quietly leaving you with no interests. The classes below
    inherit from this one, so they all get the same rule.
    """

    model_config = ConfigDict(extra="forbid")


class Interest(StrictModel):
    """One [[interests]] block: a kind of work you'd enjoy."""

    name: str  # a short label, e.g. "pipelines"
    text: str  # a sentence or two, compared against job descriptions


class AvoidItem(StrictModel):
    """One [[avoid]] block: something you don't want in a job."""

    text: str


class Filters(StrictModel):
    """The [filters] section: hard rules. Jobs that fail are dropped.

    Every field has a default, so the whole section is optional.
    """

    allow_remote: bool = True
    locations: list[str] = []
    min_salary: int | None = None  # yearly US dollars; None means no minimum
    keep_if_salary_missing: bool = True


class Weights(StrictModel):
    """The optional [weights] section: how much each part of the score counts.

    matching.py scores a job as:
        title * (title similarity) + interests * (best interest match)
        - avoid * (closest avoid match)
    These defaults are a starting point. Once the eval exists, change them
    and compare the eval's numbers before and after (see CLAUDE.md).
    """

    title: float = 1.0
    interests: float = 1.0
    avoid: float = 0.5


class Profile(StrictModel):
    """Everything in profile.toml."""

    # `Field(min_length=1)` means the list must have at least one item.
    titles: list[str] = Field(min_length=1)
    interests: list[Interest] = Field(min_length=1)
    avoid: list[AvoidItem] = []
    filters: Filters = Filters()
    weights: Weights = Weights()


class Resume(BaseModel):
    """One resume file."""

    name: str  # the file name without its extension, e.g. "data-engineer"
    text: str  # everything in the file


def profile_dir() -> Path:
    """The profile folder, from the JOBS_SEARCH_PROFILE_DIR setting (default `profile`)."""
    return Path(os.environ.get("JOBS_SEARCH_PROFILE_DIR", "profile"))


def load_profile(folder: Path) -> Profile:
    """Read `profile.toml` from `folder` and check it.

    Raises:
        FileNotFoundError: if there's no profile.toml in the folder.
        tomllib.TOMLDecodeError: if the TOML has a syntax mistake (it names the line).
        pydantic.ValidationError: if a field is missing, misspelled, or the wrong type.
    """
    path = folder / "profile.toml"
    # "rb" opens the file as raw bytes, which is what tomllib wants.
    with open(path, "rb") as file:
        data = tomllib.load(file)  # TOML text -> Python dicts and lists

    # `model_validate` builds a Profile from the dict, checking every field.
    return Profile.model_validate(data)


def load_resumes(folder: Path) -> list[Resume]:
    """Read every resume in `folder/resumes/`, sorted by file name.

    Returns an empty list if there's no resumes folder.
    Files whose names start with "." (hidden files) are skipped.

    Raises:
        ValueError: for a file that isn't .md or .txt, so a resume never gets
            silently ignored.
    """
    resumes_folder = folder / "resumes"
    if not resumes_folder.exists():
        return []

    resumes = []
    # `.iterdir()` lists everything in the folder; `sorted` puts it in name order.
    for path in sorted(resumes_folder.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue

        # `.suffix` is the file extension, e.g. ".md". `.lower()` so ".MD" counts too.
        if path.suffix.lower() not in RESUME_TYPES:
            raise ValueError(
                f"Can't read resume {path.name!r}. "
                f"Save it as one of {RESUME_TYPES} (copy the text into a new file)."
            )

        text = path.read_text(encoding="utf-8").strip()
        # `.stem` is the file name without its extension.
        resumes.append(Resume(name=path.stem, text=text))
    return resumes

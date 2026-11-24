"""The shared data shapes used across the whole project.

Every job board (Greenhouse, Lever, Ashby) returns job postings in its own
JSON format. Each fetcher translates that JSON into the `Job` class below,
so every other module only ever has to understand one shape.
"""

# `datetime` is Python's built-in type for a date plus a time of day.
from datetime import datetime

# `Literal` lets a type hint say "only these exact values are allowed".
from typing import Literal

# `BaseModel` is Pydantic's starting point for a data class. Any class that
# inherits from it (writes `(BaseModel)` after its name) gets automatic
# checking: when you create an object, Pydantic compares each value with its
# type hint and raises a clear error if it doesn't fit.
from pydantic import BaseModel

# The job boards we support. Giving this a name means we can reuse it in other
# modules (e.g. `from jobs_search.models import Source`) instead of retyping
# the list. To support a new board later, add its name here.
Source = Literal["greenhouse", "lever", "ashby", "workable", "recruitee", "personio"]


class Job(BaseModel):
    """One job posting, in the same shape no matter which board it came from.

    Each line below is a field: `name: type`. Fields without `= ...` are
    required, so creating a Job without them is an error. Fields with
    `= None` are optional and default to None (Python's word for "nothing").
    """

    # A unique key for this job, in the form "<source>:<company>:<board's id>",
    # for example "greenhouse:acme:4012345". store.py uses it to make sure the
    # same job is never saved twice.
    id: str

    # Which job board this came from. Because of `Source` above, only the
    # boards listed there (like "greenhouse" or "lever") are accepted.
    source: Source

    # The company's name as the job board knows it, for example "acme".
    company: str

    # The job title, for example "Senior Data Engineer".
    title: str

    # Where the job is, as the posting writes it, for example
    # "Minneapolis, MN" or "Remote - US". The location filter reads this.
    location: str

    # Link to the posting, so you can open it and apply.
    url: str

    # The full job description as plain text (fetchers strip out any HTML).
    # This is the text that gets embedded and compared with your interests.
    description: str

    # Yearly salary range in US dollars. `int | None` means "a whole number,
    # or nothing". Many postings don't list a salary, so these stay None
    # unless the fetcher finds one.
    salary_min: int | None = None
    salary_max: int | None = None

    # When the job was first posted. Used for questions like "what's new in
    # the last 7 days?". All three boards provide this. None if missing.
    posted_at: datetime | None = None

    # When the job board last changed this posting. Only Greenhouse provides
    # this, so it's None for Lever and Ashby jobs.
    updated_at: datetime | None = None

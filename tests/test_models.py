"""Tests for models.py.

pytest finds this file because its name starts with `test_`, and runs every
function in it whose name starts with `test_`. Inside a test, `assert` means
"this must be true". If it isn't, the test fails and pytest shows you why.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from jobs_search.models import Job


def make_job_data() -> dict:
    """Return the fields for a valid job, as a dictionary.

    A dictionary (`dict`) is a set of name -> value pairs. Each test copies
    this and changes one thing, so it's clear what that test is checking.
    """
    return {
        "id": "greenhouse:acme:12345",
        "source": "greenhouse",
        "company": "acme",
        "title": "Data Engineer",
        "location": "Minneapolis, MN",
        "url": "https://boards.greenhouse.io/acme/jobs/12345",
        "description": "Build data pipelines in Python and SQL.",
    }


def test_valid_job_is_created():
    # `**data` unpacks the dictionary into named arguments, so this is the
    # same as writing Job(id="greenhouse:acme:12345", source="greenhouse", ...).
    data = make_job_data()
    data["salary_min"] = 100000
    data["updated_at"] = "2026-11-20T09:30:00"

    job = Job(**data)

    assert job.title == "Data Engineer"
    assert job.salary_min == 100000
    # Pydantic turned the text date into a real datetime object for us.
    assert job.updated_at == datetime(2026, 11, 20, 9, 30)


def test_optional_fields_default_to_none():
    # The example data has no salary or date, so those fields should be None.
    job = Job(**make_job_data())

    assert job.salary_min is None
    assert job.salary_max is None
    assert job.updated_at is None


def test_unknown_source_is_rejected():
    data = make_job_data()
    data["source"] = "indeed"  # not one of greenhouse / lever / ashby

    # `pytest.raises` means: the code inside this block MUST raise this error.
    # If it doesn't, the test fails.
    with pytest.raises(ValidationError):
        Job(**data)


def test_missing_required_field_is_rejected():
    data = make_job_data()
    del data["title"]  # remove a required field

    with pytest.raises(ValidationError):
        Job(**data)

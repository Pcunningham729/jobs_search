"""Tests for fetchers/helpers.py."""

from jobs_search.fetchers.helpers import build_location, html_to_text


def test_html_to_text_strips_tags():
    html = "<p>Build <b>pipelines</b></p>\n\n<p>  Ship it  </p>"

    assert html_to_text(html) == "Build\npipelines\nShip it"


def test_build_location_joins_places():
    assert build_location(["Berlin", "Paris"], is_remote=False) == "Berlin; Paris"


def test_build_location_adds_remote_when_missing():
    location = build_location(["New York, NY", "Austin, TX"], is_remote=True)

    assert location == "Remote; New York, NY; Austin, TX"


def test_build_location_does_not_add_remote_twice():
    # "Remote (US)" already says remote, so nothing is added.
    assert build_location(["Remote (US)"], is_remote=True) == "Remote (US)"


def test_build_location_skips_blanks_and_duplicates():
    location = build_location(["Berlin", "", " Berlin ", "Paris"], is_remote=False)

    assert location == "Berlin; Paris"

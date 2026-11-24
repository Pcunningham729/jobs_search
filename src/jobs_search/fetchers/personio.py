"""Fetch job postings from Personio and turn them into `Job` objects.

Personio is an HR system popular with German and other European companies.
Each company gets its own subdomain, and a public feed that needs no login:

    https://<company>.jobs.personio.de/xml?language=en

Same two-step pattern as greenhouse.py:
- `parse_personio()` translates text we already have into `Job` objects (no network).
- `fetch_personio()` downloads the feed, then hands it to `parse_personio()`.

How Personio differs from the others:
- The feed is XML, not JSON. XML is an older data format made of nested tags,
  a lot like HTML:
      <workzag-jobs>
        <position>
          <id>2690964</id>
          <name>Senior AI Engineer</name>
          ...
        </position>
      </workzag-jobs>
  Python's built-in `xml.etree.ElementTree` module reads it, so we need no
  extra library.
- The description is split into named sections, and many jobs have none.
- The feed has no link to each job, so we build it from the job's id.
- No salary information.
"""

# ElementTree reads XML into a tree of "elements" (one per tag). `ET` is the
# usual short name for it.
import xml.etree.ElementTree as ET

import httpx

from jobs_search.fetchers.helpers import build_location, html_to_text
from jobs_search.models import Job

API_URL = "https://{company}.jobs.personio.de/xml"
JOB_URL = "https://{company}.jobs.personio.de/job/{job_id}"
TIMEOUT_SECONDS = 30


def element_text(parent: ET.Element, tag: str) -> str:
    """The text inside `parent`'s child <tag>, or "" if there's no such child.

    Example: for <position><name>Engineer</name></position>,
        element_text(position, "name") -> "Engineer"
    """
    # `.find(tag)` looks only at direct children, and returns None if missing.
    child = parent.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def description_to_text(position: ET.Element) -> str:
    """Join a job's description sections into plain text.

    Personio sends sections like:
        <jobDescriptions>
          <jobDescription>
            <name>YOUR IMPACT</name>
            <value><![CDATA[<ul><li>Build AI solutions</li></ul>]]></value>
          </jobDescription>
          ...
        </jobDescriptions>
    (`<![CDATA[ ... ]]>` just means "the text in here is raw, don't read it as
    XML". Here it holds HTML.) Each section becomes its heading followed by
    its text. Returns "" if the job has no sections.
    """
    html_parts = []
    # `.iter(tag)` finds every <jobDescription>, however deeply nested.
    for section in position.iter("jobDescription"):
        heading = element_text(section, "name")
        body = element_text(section, "value")
        html_parts.append(f"<p>{heading}</p>{body}")
    return html_to_text("".join(html_parts))


def parse_personio(xml_text: str, company: str) -> list[Job]:
    """Turn a Personio XML feed into a list of `Job` objects.

    `xml_text` is the feed exactly as downloaded (text, not JSON).
    `company` is the company's Personio name, for example "ottonova".
    """
    # `fromstring` reads the whole feed. `root` is the outer <workzag-jobs> tag.
    root = ET.fromstring(xml_text)

    jobs = []
    for position in root.findall("position"):
        job_id = element_text(position, "id")

        # The main office, then any extra ones inside <additionalOffices>.
        places = [element_text(position, "office")]
        additional = position.find("additionalOffices")
        if additional is not None:
            for office in additional.findall("office"):
                places.append((office.text or "").strip())

        job = Job(
            id=f"personio:{company}:{job_id}",
            source="personio",
            company=company,
            title=element_text(position, "name"),
            # Personio has no remote field. Remote jobs say so in the office name.
            location=build_location(places, is_remote=False),
            url=JOB_URL.format(company=company, job_id=job_id),
            description=description_to_text(position),
            salary_min=None,
            salary_max=None,
            # Text like "2026-11-13T16:21:35+00:00". Pydantic turns it into a
            # datetime. `or None` turns a missing date ("") into None.
            posted_at=element_text(position, "createdAt") or None,
            updated_at=None,
        )
        jobs.append(job)
    return jobs


def fetch_personio(company: str) -> list[Job]:
    """Download all open jobs for one company from Personio.

    `company` is the company's Personio name: the part before
    `.jobs.personio.de` in its careers address, for example "ottonova".
    """
    url = API_URL.format(company=company)

    # language=en asks for the English version of each job, when there is one.
    response = httpx.get(url, params={"language": "en"}, timeout=TIMEOUT_SECONDS)

    # Stop with a clear error if Personio answered with one (e.g. 404 for a wrong name).
    response.raise_for_status()

    # `.text` (not `.json()`): the feed is XML text.
    return parse_personio(response.text, company)

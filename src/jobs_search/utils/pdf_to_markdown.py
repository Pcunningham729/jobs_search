"""Convert a PDF (like a resume) into a Markdown file you can read and edit.

The job agent only reads .md and .txt resumes, because PDF text is hard to
get out reliably. This tool does the conversion once, so you can check the
result and fix anything that came out wrong.

Run it from the repo root:

    uv run python -m jobs_search.utils.pdf_to_markdown "path/to/resume.pdf"

It writes "path/to/resume.md" next to the PDF. It won't overwrite an
existing .md file, so edits you've made are never lost.

ALWAYS read the result. PDFs store text as positioned pieces, not as
paragraphs, so two-column layouts can come out interleaved, and bullets or
special characters can come out oddly.

(`python -m` runs a module inside a package as a program.)
"""

import re
import sys
from pathlib import Path

# pypdf is a library for reading PDF files.
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Tidy text pulled out of a PDF.

    - removes spaces at the end of each line
    - turns runs of 3 or more line breaks into one blank line
    - removes blank lines at the very start and end
    """
    lines = []
    for line in text.splitlines():
        lines.append(line.rstrip())  # `rstrip` = strip from the right end only
    text = "\n".join(lines)

    # `re` is Python's built-in "regular expressions" module: a mini-language
    # for finding patterns in text. "\n{3,}" means "3 or more line breaks
    # in a row"; we replace each run with exactly 2 (one blank line).
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def pdf_to_text(pdf_path: Path) -> str:
    """Pull the text out of every page of a PDF, with a blank line between pages."""
    reader = PdfReader(pdf_path)

    page_texts = []
    for page in reader.pages:
        # `or ""`: a page with no text (e.g. a scanned image) gives back None.
        page_texts.append(page.extract_text() or "")

    return clean_text("\n\n".join(page_texts))


def convert_pdf(pdf_path: Path) -> Path:
    """Convert `pdf_path` to a .md file in the same folder. Returns the new file's path.

    Raises:
        FileNotFoundError: if the PDF doesn't exist.
        FileExistsError: if the .md file already exists (so your edits are safe).
        ValueError: if no text could be found, e.g. a scanned image of a resume.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"No such file: {pdf_path}")

    # `.with_suffix(".md")` swaps the extension: resume.pdf -> resume.md
    md_path = pdf_path.with_suffix(".md")
    if md_path.exists():
        raise FileExistsError(
            f"{md_path} already exists. Delete or rename it first if you "
            "really want to convert again."
        )

    text = pdf_to_text(pdf_path)
    if not text.strip():
        raise ValueError(
            f"No text found in {pdf_path.name}. It may be a scanned image; "
            "if so, type or copy the text into a .md file by hand."
        )

    md_path.write_text(text, encoding="utf-8")
    return md_path


def main() -> None:
    """Run from the command line: convert each PDF path given after the command."""
    # `sys.argv` is the list of words typed on the command line. Item 0 is
    # the program itself, so the file paths start at item 1.
    pdf_paths = sys.argv[1:]
    if not pdf_paths:
        print('Usage: uv run python -m jobs_search.utils.pdf_to_markdown "file.pdf"')
        sys.exit(1)  # a non-zero exit code tells the terminal something went wrong

    for name in pdf_paths:
        md_path = convert_pdf(Path(name))
        print(f"Wrote {md_path}. Read it and fix anything that came out jumbled.")


# This is true only when the file is run as a program (with `python -m ...`),
# not when another module imports it. So importing it never converts anything.
if __name__ == "__main__":
    main()

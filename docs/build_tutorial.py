"""Build docs/tutorial.pdf: a guided tour of how jobs_search works.

Run from the repo root:

    uv run --with reportlab python docs/build_tutorial.py

`--with reportlab` borrows the reportlab PDF library just for this run, so it
isn't added to the project's dependencies (the app itself never needs it).
Edit the text below and rerun to update the PDF.
"""

import math
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path(__file__).parent / "tutorial.pdf"
PAGE_WIDTH = letter[0] - 2 * inch  # usable width with 1-inch margins

# ---------------------------------------------------------------------------
# Colours and text styles
# ---------------------------------------------------------------------------

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
LINE = colors.HexColor("#9aa5b1")
COLLECT = colors.HexColor("#e3f0fb")  # collecting jobs (no LLM)
USE = colors.HexColor("#e6f4ea")  # scoring with embeddings (no LLM)
LLM = colors.HexColor("#fde8d7")  # steps where Claude runs
DATA = colors.HexColor("#f2f2f2")  # files and databases
HEADER = colors.HexColor("#334e68")

base = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
    leading=14, textColor=INK, spaceAfter=6,
)  # fmt: skip
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CAPTION = ParagraphStyle(
    "caption", parent=BODY, fontSize=8.5, textColor=MUTED, alignment=TA_CENTER,
    spaceBefore=2, spaceAfter=12,
)  # fmt: skip
H1 = ParagraphStyle(
    "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18,
    textColor=HEADER, spaceBefore=4, spaceAfter=10,
)  # fmt: skip
H2 = ParagraphStyle(
    "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
    textColor=HEADER, spaceBefore=10, spaceAfter=6,
)  # fmt: skip
H3 = ParagraphStyle(
    "h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.5,
    textColor=INK, spaceBefore=8, spaceAfter=3,
)  # fmt: skip
CODE = ParagraphStyle(
    "code", fontName="Courier", fontSize=8.5, leading=11, textColor=INK,
    backColor=colors.HexColor("#f5f7fa"), borderPadding=6, leftIndent=6,
    rightIndent=6, spaceBefore=4, spaceAfter=10,
)  # fmt: skip
NOTE = ParagraphStyle(
    "note", parent=BODY, backColor=colors.HexColor("#fff8e1"), borderPadding=7,
    leftIndent=7, rightIndent=7, spaceBefore=6, spaceAfter=12,
)  # fmt: skip


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    """A paragraph. Text uses reportlab's mini-HTML: <b>, <i>, <font>."""
    return Paragraph(text, style)


def c(text: str) -> str:
    """Inline code: monospace, with < > & made safe for the mini-HTML."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<font face="Courier">{safe}</font>'


def bullets(items: list[str]) -> list[Paragraph]:
    style = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=4,
                           spaceAfter=3)  # fmt: skip
    return [Paragraph(item, style, bulletText="•") for item in items]


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


def table(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
    """A table whose cells wrap. Widths are fractions of the page width."""
    cells = []
    for row_number, row in enumerate(rows):
        style = SMALL
        if header and row_number == 0:
            style = ParagraphStyle("th", parent=SMALL, fontName="Helvetica-Bold",
                                   textColor=colors.white)  # fmt: skip
        cells.append([Paragraph(cell, style) for cell in row])
    result = Table(cells, colWidths=[w * PAGE_WIDTH for w in widths], repeatRows=1)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), HEADER))
        for row_number in range(2, len(rows), 2):
            commands.append(("BACKGROUND", (0, row_number), (-1, row_number),
                             colors.HexColor("#f7f9fb")))  # fmt: skip
    result.setStyle(TableStyle(commands))
    return result


def figure(drawing: Drawing, caption: str) -> KeepTogether:
    return KeepTogether([drawing, p(caption, CAPTION)])


# ---------------------------------------------------------------------------
# Drawing helpers for the diagrams
# ---------------------------------------------------------------------------


def box(d: Drawing, x, y, w, h, lines: list[str], fill, bold_first=True, size=8):
    """A rounded box with centred lines of text. (x, y) is the bottom-left."""
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=LINE,
               strokeWidth=0.8))  # fmt: skip
    gap = size + 2.5
    top = y + h / 2 + (len(lines) - 1) * gap / 2 - size / 3
    for number, text in enumerate(lines):
        font = "Helvetica-Bold" if (bold_first and number == 0) else "Helvetica"
        colour = INK if number == 0 else MUTED
        d.add(String(x + w / 2, top - number * gap, text, fontName=font,
                     fontSize=size if number == 0 else size - 0.5,
                     fillColor=colour, textAnchor="middle"))  # fmt: skip


def arrow(d: Drawing, x1, y1, x2, y2, label: str = "", dashed=False, label_dy=4):
    """A line with an arrowhead at (x2, y2), and an optional label."""
    d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1,
               strokeDashArray=[3, 2] if dashed else None))  # fmt: skip
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 5
    left = (x2 - size * math.cos(angle - 0.4), y2 - size * math.sin(angle - 0.4))
    right = (x2 - size * math.cos(angle + 0.4), y2 - size * math.sin(angle + 0.4))
    d.add(Polygon([x2, y2, *left, *right], fillColor=MUTED, strokeColor=MUTED))
    if label:
        d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label,
                     fontName="Helvetica-Oblique", fontSize=6.5, fillColor=MUTED,
                     textAnchor="middle"))  # fmt: skip


def lane_label(d: Drawing, x, y, text: str):
    d.add(String(x, y, text, fontName="Helvetica-Bold", fontSize=7.5,
                 fillColor=HEADER))  # fmt: skip


def legend(d: Drawing, x, y):
    items = [(COLLECT, "collect (no LLM)"), (USE, "embeddings + scoring (no LLM)"),
             (LLM, "Claude (LLM) runs here"), (DATA, "files / databases")]  # fmt: skip
    for fill, text in items:
        d.add(Rect(x, y, 9, 9, fillColor=fill, strokeColor=LINE, strokeWidth=0.6))
        d.add(String(x + 13, y + 1.5, text, fontName="Helvetica", fontSize=7,
                     fillColor=MUTED))  # fmt: skip
        x += 13 + len(text) * 3.6 + 14


# ---------------------------------------------------------------------------
# The diagrams
# ---------------------------------------------------------------------------


def note(d: Drawing, x, y, text: str, anchor: str = "middle", size: float = 6.5):
    """A small italic label, placed by hand so it never sits on a line."""
    d.add(String(x, y, text, fontName="Helvetica-Oblique", fontSize=size,
                 fillColor=MUTED, textAnchor=anchor))  # fmt: skip


def diagram_big_picture() -> Drawing:
    w, h = PAGE_WIDTH, 340
    d = Drawing(w, h)
    bw, bh = 98, 46
    gap = (w - 4 * bw) / 3
    xs = [i * (bw + gap) for i in range(4)]

    lane_label(d, 0, 322, "1. COLLECT   (jobs-search discover, jobs-search fetch)")
    y1 = 262
    box(
        d,
        xs[0],
        y1,
        bw,
        bh,
        ["discovery.py", "finds job boards", "-> companies.csv"],
        COLLECT,
    )
    box(d, xs[1], y1, bw, bh, ["fetchers/", "board APIs", "-> list[Job]"], COLLECT)
    box(d, xs[2], y1, bw, bh, ["store.py", "SQLite: every job,", "your statuses"], DATA)
    box(d, xs[3], y1, bw, bh, ["rag.py", "+ embeddings.py", "vectors in Chroma"], DATA)
    for i in range(3):
        arrow(d, xs[i] + bw, y1 + bh / 2, xs[i + 1], y1 + bh / 2)
    note(d, xs[1] + bw / 2, y1 - 11, "pipeline.py runs fetch -> store -> index")

    lane_label(d, 0, 212, "2. USE   (jobs-search match, review, chat)")
    y2 = 150
    box(d, xs[0], y2, bw, bh, ["profile.py", "profile.toml", "+ resumes/"], DATA)
    box(
        d, xs[1], y2, bw, bh, ["matching.py", "filters + scores", "-> ranked jobs"], USE
    )
    box(d, xs[2], y2, bw, bh, ["review.py", "Claude reranks", "the top 20"], LLM)
    box(d, xs[3], y2, bw, bh, ["main.py", "prints results", "for you"], USE)
    for i in range(3):
        arrow(d, xs[i] + bw, y2 + bh / 2, xs[i + 1], y2 + bh / 2)
    # The databases feed matching.
    arrow(d, xs[2] + bw * 0.3, y1, xs[1] + bw * 0.7, y2 + bh)
    arrow(d, xs[3] + bw * 0.3, y1, xs[1] + bw * 0.92, y2 + bh)
    note(d, xs[2] + 38, 238, "jobs + stored vectors")

    y3 = 40
    box(d, xs[1], y3, bw * 2 + gap, bh, ["agent.py + tools.py",
        "Claude chats with you and calls tools:",
        "find_matches, get_job_details, set_status"], LLM)  # fmt: skip
    box(
        d,
        xs[0],
        y3,
        bw,
        bh,
        ["prompts.py", "agent_prompts/*.md", "-> system prompt"],
        DATA,
    )
    box(d, xs[3], y3, bw, bh, ["you", "jobs-search chat"], USE)
    arrow(d, xs[0] + bw, y3 + bh / 2, xs[1], y3 + bh / 2)
    arrow(d, xs[3], y3 + bh / 2 + 6, xs[2] + bw, y3 + bh / 2 + 6)
    arrow(d, xs[2] + bw, y3 + bh / 2 - 6, xs[3], y3 + bh / 2 - 6)
    # find_matches -> matching.py
    arrow(d, xs[1] + bw / 2, y3 + bh, xs[1] + bw / 2, y2)
    note(d, xs[1] + bw / 2 + 4, 116, "find_matches", anchor="start")
    # get_job_details / set_status -> store.py, routed between the columns
    route_x = xs[2] + bw + gap / 2
    arrow(d, route_x, y3 + bh, xs[2] + bw - 4, y1, dashed=True)
    note(d, route_x + 5, 104, "get_job_details,", anchor="start")
    note(d, route_x + 5, 96, "set_status -> store", anchor="start")

    legend(d, 0, 4)
    return d


def diagram_fetch() -> Drawing:
    w, h = PAGE_WIDTH, 150
    d = Drawing(w, h)
    bw, bh = 80, 44
    gap = (w - 5 * bw) / 4
    xs = [i * (bw + gap) for i in range(5)]
    y = 78
    box(d, xs[0], y, bw, bh, ["companies.csv", "-> list[Company]"], DATA)
    box(d, xs[1], y, bw, bh, ["fetcher(slug)", "-> list[Job]"], COLLECT)
    box(
        d,
        xs[2],
        y,
        bw,
        bh,
        ["save_jobs()", "-> SaveResult:", "new + changed ids"],
        DATA,
    )
    box(
        d,
        xs[3],
        y,
        bw,
        bh,
        ["jobs_to_index()", "passes filters,", "new or changed"],
        USE,
    )
    box(d, xs[4], y, bw, bh, ["index_jobs()", "chunk + embed", "-> Chroma"], DATA)
    for i in range(4):
        arrow(d, xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)
    note(d, 0, 138, "fetch_and_index() repeats this row for every company; one failure "
         "never stops the run", anchor="start", size=7)  # fmt: skip

    box(d, xs[1], 10, bw, 36, ["HTTP 404", "board closed"], DATA)
    arrow(d, xs[1] + bw / 2, y, xs[1] + bw / 2, 46)
    box(d, xs[0], 10, bw, 36, ["remove_companies()", "off the list"], COLLECT, size=7)
    arrow(d, xs[1], 28, xs[0] + bw, 28)
    note(
        d,
        xs[2],
        34,
        "Every job is saved in SQLite, but only jobs that",
        anchor="start",
        size=7,
    )
    note(
        d,
        xs[2],
        24,
        "pass your filters are embedded (the slow part).",
        anchor="start",
        size=7,
    )
    return d


def diagram_index_vs_query() -> Drawing:
    w, h = PAGE_WIDTH, 250
    d = Drawing(w, h)
    lane_label(d, 0, 236, "INDEX TIME  (once per new or changed job, during fetch)")
    bw, bh = 92, 40
    gap = (w - 4 * bw) / 3
    xs = [i * (bw + gap) for i in range(4)]
    y = 170
    box(d, xs[0], y, bw, bh, ["Job.description", "plain text"], DATA)
    box(d, xs[1], y, bw, bh, ["chunk_text()", "pieces <= 1,000 chars"], USE)
    box(d, xs[2], y, bw, bh, ["embed_texts()", "bge-small: 384 numbers"], USE)
    box(d, xs[3], y, bw, bh, ["Chroma", "job_chunks collection"], DATA)
    arrow(d, xs[0] + bw, y + bh / 2, xs[1], y + bh / 2, "str")
    arrow(d, xs[1] + bw, y + bh / 2, xs[2], y + bh / 2, "list[str]")
    arrow(d, xs[2] + bw, y + bh / 2, xs[3], y + bh / 2, "vectors")
    box(d, xs[2], y - 52, bw, 34, ["Job.title -> 1 vector"], USE)
    box(d, xs[3], y - 52, bw, 34, ["job_titles collection"], DATA)
    arrow(d, xs[2] + bw, y - 35, xs[3], y - 35)

    lane_label(d, 0, 100, "QUERY TIME  (every match, review or find_matches call)")
    y = 30
    box(d, xs[0], y, bw, bh, ["Profile", "titles, interests, avoid"], DATA)
    box(d, xs[1], y, bw, bh, ["embed_profile()", "-> ProfileVectors"], USE)
    box(d, xs[2], y, bw, bh, ["score_job()", "cosine similarity", "pure maths"], USE)
    box(d, xs[3], y, bw, bh, ["Scores", "title, interest,", "avoid, total"], USE)
    arrow(d, xs[0] + bw, y + bh / 2, xs[1], y + bh / 2)
    arrow(d, xs[1] + bw, y + bh / 2, xs[2], y + bh / 2)
    arrow(d, xs[2] + bw, y + bh / 2, xs[3], y + bh / 2)
    # Stored vectors come back out of Chroma for scoring.
    arrow(d, xs[3] + bw * 0.2, 118, xs[2] + bw * 0.8, y + bh)
    note(d, xs[3] + bw * 0.2 + 2, 92, "stored vectors", anchor="start")
    note(d, xs[3] + bw * 0.2 + 2, 84, "(get_job_vectors)", anchor="start")
    return d


def diagram_best_chunk() -> Drawing:
    w, h = PAGE_WIDTH, 150
    d = Drawing(w, h)
    box(d, 10, 55, 120, 40, ["interest: pipelines", "'Building reliable data",
                             "pipelines...' -> vector"], USE)  # fmt: skip
    chunks = [("chunk 0: company pitch", "0.48"), ("chunk 1: what you'll do", "0.69"),
              ("chunk 2: benefits", "0.44")]  # fmt: skip
    for number, (text, score) in enumerate(chunks):
        y = 110 - number * 45
        box(d, 230, y, 130, 30, [text], DATA, bold_first=False)
        arrow(d, 130, 75, 230, y + 15, score, label_dy=3)
    box(d, w - 92, 55, 92, 40, ["best = 0.69", "max over chunks"], USE)
    arrow(d, 360, 80, w - 92, 75)
    return d


def diagram_review() -> Drawing:
    w, h = PAGE_WIDTH, 120
    d = Drawing(w, h)
    bw, bh = 86, 46
    gap = (w - 5 * bw) / 4
    xs = [i * (bw + gap) for i in range(5)]
    y = 50
    box(d, xs[0], y, bw, bh, ["rank_jobs()", "top 20", "list[MatchResult]"], USE)
    box(
        d, xs[1], y, bw, bh, ["build_prompt()", "profile + resumes", "+ every job"], USE
    )
    box(
        d, xs[2], y, bw, bh, ["Claude", "messages.parse()", "screen-posting skill"], LLM
    )
    box(d, xs[3], y, bw, bh, ["ReviewBatch", "a JobReview", "per job_id"], LLM)
    box(d, xs[4], y, bw, bh, ["review_jobs()", "sort by fit, then", "match score"], USE)
    for i in range(4):
        arrow(d, xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)
    note(d, xs[1] + bw / 2, 38, "emails + phones removed")
    note(d, xs[3] + bw / 2, 38, "checked against the schema")
    note(d, xs[4] + bw / 2, 38, "-> list[ReviewedJob]")
    legend(d, 0, 4)
    return d


def diagram_agent_loop() -> Drawing:
    w, h = PAGE_WIDTH, 200
    d = Drawing(w, h)
    box(d, 0, 80, 90, 44, ["you", "'anything good", "this week?'"], USE)
    box(d, 150, 130, 150, 50, ["Claude (agent)", "reads system prompt + history,",
                               "decides: answer, or call a tool?"], LLM)  # fmt: skip
    box(d, 150, 20, 150, 50, ["tools.py", "find_matches / get_job_details /",
                              "set_status (runs real code)"], USE)  # fmt: skip
    box(d, w - 110, 20, 110, 50, ["matching.py, store.py", "(the real logic)"], USE)
    box(
        d, w - 110, 130, 110, 50, ["final answer", "text with job ids", "and URLs"], USE
    )
    arrow(d, 90, 110, 150, 150)
    note(d, 58, 132, "question")
    arrow(d, 205, 130, 205, 70)
    note(d, 200, 98, "tool call", anchor="end")
    arrow(d, 245, 70, 245, 130)
    note(d, 250, 98, "tool result (text)", anchor="start")
    arrow(d, 300, 45, w - 110, 45)
    arrow(d, 300, 160, w - 110, 160)
    note(d, (300 + w - 110) / 2, 166, "no more tools")
    note(d, 150, 4, "create_agent() repeats call -> result -> Claude until Claude answers "
         "in plain text", anchor="start", size=7)  # fmt: skip
    return d


# ---------------------------------------------------------------------------
# The content
# ---------------------------------------------------------------------------


def title_page() -> list:
    big = ParagraphStyle("title", parent=H1, fontSize=28, leading=34, spaceAfter=14)
    return [
        Spacer(1, 1.6 * inch),
        p("jobs_search", big),
        p(
            "How the job-search agent works",
            ParagraphStyle("sub", parent=H2, fontSize=16),
        ),
        Spacer(1, 10),
        p(
            "A tutorial on the pipeline, how the modules hand data to each other, the "
            "RAG and embedding components, and exactly what Claude (the LLM) does."
        ),
        Spacer(1, 30),
        p("<b>How to read this</b>"),
        *bullets(
            [
                (
                    "<b>Part 1</b> is the big picture: what runs when, and what each module takes in "
                    "and gives back."
                ),
                "<b>Part 2</b> zooms in on embeddings and RAG, with real numbers from this project.",
                "<b>Part 3</b> is about the LLM: the two places Claude runs, and why only there.",
                "<b>Part 4</b> is a module-by-module reference of the classes and functions.",
                "<b>Part 5</b> covers testing and the eval. A glossary is at the end.",
            ]
        ),
        Spacer(1, 30),
        p("Box colours in every diagram mean the same thing:", SMALL),
        _legend_only(),
        Spacer(1, 40),
        p(
            "Rebuild this PDF after changing the code: "
            + c("uv run --with reportlab python docs/build_tutorial.py"),
            SMALL,
        ),
        PageBreak(),
    ]


def _legend_only() -> Drawing:
    d = Drawing(PAGE_WIDTH, 16)
    legend(d, 0, 3)
    return d


def part1() -> list:
    io_rows = [
        ["Module", "Main function", "Takes in", "Gives back", "Used next by"],
        ["discovery.py", c("discover()"), "candidate " + c("set[Company]") + " (from Common Crawl and "
         + c("company_seeds.txt") + ")", c("DiscoveryReport") + "; writes " + c("companies.csv"),
         "pipeline.py reads " + c("companies.csv")],
        ["fetchers/", c("fetch_greenhouse()") + ", " + c("fetch_lever()") + ", " + c("fetch_ashby()"), "a company slug, e.g. " + c('"gitlab"'),
         c("list[Job]"), "pipeline.py"],
        ["store.py", c("save_jobs()"), c("list[Job]"),
         c("SaveResult") + ": " + c("new_ids") + ", " + c("changed_ids"), "pipeline.py decides what to index"],
        ["rag.py", c("index_jobs()"), c("list[Job]") + " + an embed function",
         "vectors stored in Chroma (returns chunk count)", "matching.py reads stored vectors"],
        ["embeddings.py", c("embed_texts()"), c("list[str]"), c("list[list[float]]") + " (384 numbers each)",
         "rag.py (jobs), matching.py (profile)"],
        ["profile.py", c("load_profile()"), c("profile/profile.toml"), c("Profile"),
         "matching, review, agent"],
        ["profile.py", c("load_resumes()"), c("profile/resumes/*.md"), c("list[Resume]"), "review.py"],
        ["matching.py", c("rank_jobs()"), "the store, the index, a " + c("Profile"),
         c("list[MatchResult]") + ", best first", "main.py, review.py, tools.py"],
        ["review.py", c("review_jobs()"), c("list[MatchResult]") + " + profile + resumes",
         c("list[ReviewedJob]") + " sorted by Claude's fit", "main.py prints it"],
        ["prompts.py", c("agent_system_prompt()"), c("agent_prompts/*.md"), "one " + c("str"), "agent.py"],
        ["tools.py", c("make_tools()"), c("ToolContext"), "3 LangChain tools", "agent.py"],
        ["agent.py", c("ask()"), "your question + the conversation",
         "answer text + updated history", "main.py chat loop"],
        ["evaluation.py", c("evaluate()"), "your labels " + c("list[LabeledJob]"), c("EvalReport") + " (precision@10)",
         "evals/rank_eval.py"],
    ]  # fmt: skip

    return [
        p("Part 1. The big picture", H1),
        p(
            "The goal: every day, look at thousands of real job postings and show you the handful "
            "worth your time, with honest reasons. The program does that in four moves:"
        ),
        *bullets(
            [
                "<b>Collect.</b> Find companies with public job boards, download their postings, and "
                "save them. ("
                + c("discovery.py")
                + ", "
                + c("fetchers/")
                + ", "
                + c("store.py")
                + ")",
                "<b>Index.</b> Turn each posting into <i>embeddings</i> (lists of numbers that capture "
                "meaning) so they can be compared with your interests by maths, not keywords. ("
                + c("embeddings.py")
                + ", "
                + c("rag.py")
                + ")",
                "<b>Score.</b> Drop jobs that fail your hard filters (place, salary), then score the rest "
                "against your profile and rank them. (" + c("matching.py") + ")",
                "<b>Judge and talk.</b> Claude reads the best few next to your resume and reranks them "
                "with reasons, or chats with you about them. ("
                + c("review.py")
                + ", "
                + c("agent.py")
                + ")",
            ]
        ),
        figure(
            diagram_big_picture(),
            "Figure 1. The modules and how data flows between them. The top row fills the "
            "databases; the bottom rows read them.",
        ),
        p("The two halves", H2),
        p(
            "The <b>collect</b> half runs when you type "
            + c("jobs-search fetch")
            + ". It is slow "
            "(network and embedding) but needs no LLM and costs nothing. The <b>use</b> half runs "
            "whenever you ask for results: "
            + c("match")
            + " is instant and free, while "
            + c("review")
            + " and "
            + c("chat")
            + " call Claude and cost a little per run."
        ),
        table(
            [
                ["Command", "What runs", "Claude?", "Cost"],
                [
                    c("jobs-search discover"),
                    "discovery.py: collect candidate boards, check each against its API",
                    "no",
                    "free",
                ],
                [
                    c("jobs-search fetch"),
                    "pipeline.py: fetchers -> store -> rag/embeddings",
                    "no",
                    "free",
                ],
                [
                    c("jobs-search match"),
                    "matching.rank_jobs(), printed with tools.format_match()",
                    "no",
                    "free",
                ],
                [
                    c("jobs-search review"),
                    "rank_jobs() top 20 -> review.review_jobs()",
                    "yes, once",
                    "a few cents",
                ],
                [
                    c("jobs-search chat"),
                    "agent.py + tools.py in a loop",
                    "yes, every turn",
                    "a little per message",
                ],
            ],
            [0.22, 0.52, 0.12, 0.14],
        ),
        Spacer(1, 6),
        p("Input and output of every module", H2),
        p(
            "This is the contract between modules: what each one expects to be handed and what it "
            "hands on. Because each module only depends on these shapes, you can change how one works "
            "inside without breaking the others."
        ),
        table(io_rows, [0.16, 0.19, 0.23, 0.22, 0.20]),
        p("The fetch step in detail", H2),
        p(
            c("pipeline.fetch_and_index()")
            + " is the glue for the collect half. For each company "
            "it calls the right fetcher, saves the jobs, and indexes only what is needed:"
        ),
        figure(
            diagram_fetch(),
            "Figure 2. One pass of the fetch loop. The output of each step is the input of the next.",
        ),
        *bullets(
            [
                c("save_jobs()")
                + " compares each job's <i>content hash</i> (a fingerprint of its "
                "title, location, description and salary) with the saved one, so it can report exactly "
                "which jobs are new or changed.",
                c("jobs_to_index()")
                + " keeps jobs that pass your filters <i>and</i> are new, changed "
                "or missing from the index. On a live test, 2 boards gave 145 jobs but only 34 needed "
                "embedding.",
                "A board answering 404 has closed, so "
                + c("remove_companies()")
                + " takes it off "
                + c("companies.csv")
                + ". Other errors are just reported and retried next time.",
            ]
        ),
        p("Where the data lives", H2),
        table(
            [
                ["Place", "Holds", "Rule"],
                [
                    c("data/jobs.db") + " (SQLite)",
                    (
                        "every job ever fetched (as JSON), its content hash, "
                        "your status (new / applied / rejected / hidden), first and last seen dates"
                    ),
                    "<b>The source of truth.</b> Never delete casually.",
                ],
                [
                    c("data/chroma/") + " (Chroma)",
                    "vectors for job chunks and titles",
                    "Can always be rebuilt from SQLite ("
                    + c("rag.rebuild_index()")
                    + ").",
                ],
                [
                    c("data/models/"),
                    "the downloaded embedding model (about 67 MB)",
                    "Downloaded once, then offline.",
                ],
                [
                    c("companies.csv"),
                    "the job boards to fetch",
                    "Built by discover, pruned by fetch. In git.",
                ],
                [
                    c("profile/"),
                    "profile.toml, resumes/, labeled_jobs.csv",
                    "Personal: never in git.",
                ],
                [
                    c("agent_prompts/"),
                    "system.md and skills.md: the job agent's instructions",
                    "Edit to change Claude's behaviour; run the eval before and after.",
                ],
            ],
            [0.2, 0.5, 0.3],
        ),
        p("The shared data shape: Job", H2),
        p(
            "Every fetcher turns its board's JSON into the same Pydantic model, "
            + c("models.Job")
            + ". Everything downstream only knows about "
            + c("Job")
            + ", never about Greenhouse "
            "or Lever, which is why adding a new board means writing one new fetcher and nothing else."
        ),
        table(
            [
                ["Field", "Type", "Example / meaning"],
                [
                    c("id"),
                    c("str"),
                    c('"lever:spotify:1a2b"')
                    + " - board, company and the board's own id",
                ],
                [
                    c("source"),
                    c('Literal["greenhouse","lever","ashby"]'),
                    "which board it came from",
                ],
                [
                    c("company, title, location, url"),
                    c("str"),
                    c('"Remote; New York, NY"'),
                ],
                [
                    c("description"),
                    c("str"),
                    "plain text (HTML removed by " + c("helpers.html_to_text") + ")",
                ],
                [
                    c("salary_min, salary_max"),
                    c("int | None"),
                    "USD per year, when the board lists it",
                ],
                [
                    c("posted_at, updated_at"),
                    c("datetime | None"),
                    "used by " + c("--days") + " / " + c("new_since_days"),
                ],
            ],
            [0.3, 0.3, 0.4],
        ),
        PageBreak(),
    ]


def part2() -> list:
    return [
        p("Part 2. Embeddings and RAG", H1),
        p(
            "This part explains the two ideas that let the program compare a job with your interests "
            "by meaning: <b>embeddings</b> and <b>retrieval-augmented generation (RAG)</b>."
        ),
        p("2.1 What an embedding is", H2),
        p(
            "An <b>embedding model</b> turns a piece of text into a fixed-length list of numbers, a "
            "<b>vector</b>. It is trained so that texts with similar meanings get vectors pointing in "
            "similar directions. This project uses "
            + c("BAAI/bge-small-en-v1.5")
            + " through the "
            + c("fastembed")
            + " library: 384 numbers per text, running on your own computer, free "
            "and offline after one download."
        ),
        p(
            "To compare two vectors we use <b>cosine similarity</b>: 1.0 means the same direction "
            "(same meaning), lower means less related. Real numbers from this project's model, "
            'comparing everything with <i>"Build data pipelines."</i>:'
        ),
        table(
            [
                ["Text", "Similarity"],
                ["Design ETL workflows that move data into the warehouse.", "0.75"],
                ["Sell enterprise software to new customers.", "0.61"],
                ["Frequent travel to client sites (up to 75%).", "0.52"],
                ["excessive travel", "0.48"],
            ],
            [0.8, 0.2],
        ),
        Spacer(1, 6),
        p(
            "Notice two things. First, the ETL sentence scores highest even though it shares only the "
            'word "data": that\'s meaning, not keywords. Second, the scores are squeezed together: '
            "even unrelated texts land near 0.5 with this model. That's why your avoid score is about "
            '0.47 to 0.55 on <i>every</i> job, and why a short avoid phrase like "excessive travel" '
            "barely separates jobs (it scores 0.78 against the travel sentence, but 0.47 against "
            "unrelated ones). Rewriting it as a full sentence is a good first eval experiment."
        ),
        code("""
from jobs_search.embeddings import embed_texts
from jobs_search.matching import cosine_similarity
a, b = embed_texts(["Build data pipelines.", "Design ETL workflows."])
len(a)                      # 384
cosine_similarity(a, b)     # about 0.7
"""),
        p(
            "In code: "
            + c("embeddings.embed_texts(texts)")
            + " loads the model once (cached in "
            + c("_loaded_models")
            + ") and returns one vector per text, in order. Its type, "
            + c("Embedder = Callable[[list[str]], list[list[float]]]")
            + ", is what every other "
            "module asks for, so tests can pass a fake embedder instead of the real model."
        ),
        p("2.2 Why job descriptions are chunked", H2),
        p(
            "A job description mixes a company pitch, the actual duties, requirements and benefits. "
            "One vector for the whole thing would blur them together, and the model only reads about "
            "512 tokens (roughly 2,000 characters) anyway. So "
            + c("rag.chunk_text()")
            + " splits "
            "each description into <b>chunks</b> of at most 1,000 characters ("
            + c("CHUNK_MAX_CHARS")
            + "), keeping whole lines together; "
            + c("split_long_paragraph()")
            + " splits a single over-long line between words. Each chunk gets its own vector."
        ),
        p(
            "When scoring, each of your interests is compared with <i>every</i> chunk and keeps its "
            '<b>best</b> match, so the duties section can match "pipelines" strongly even if the '
            "benefits section doesn't:"
        ),
        figure(
            diagram_best_chunk(),
            "Figure 3. One interest against three chunks of one job. The job's score for that "
            "interest is the best chunk (illustrative numbers).",
        ),
        p("2.3 Index time and query time", H2),
        p(
            "A key rule from the architecture: <b>job vectors are computed once</b>, when a job is "
            "fetched (or changes), and stored; <b>profile vectors are computed fresh on every "
            "query</b>. Embedding thousands of job chunks is slow, but embedding your dozen profile "
            "lines takes a moment, so editing your profile takes effect immediately without "
            "re-indexing anything."
        ),
        figure(
            diagram_index_vs_query(),
            "Figure 4. Job text is embedded once and stored; the profile is embedded on every "
            "query; scoring then only does maths on stored vectors.",
        ),
        p("2.4 What is stored in Chroma", H2),
        p(
            "<b>Chroma</b> is a <i>vector database</i>: it stores vectors with ids and extra facts "
            "(<i>metadata</i>), and can find the vectors closest to a query. "
            + c("rag.open_index()")
            + " opens two <i>collections</i> (like tables), both set to cosine distance "
            "(distance = 1 - similarity):"
        ),
        table(
            [
                ["Collection", "One entry per", "Id", "Metadata", "Read by"],
                [
                    c("job_chunks"),
                    "description chunk",
                    c('"<job id>#<n>"'),
                    c("job_id, chunk"),
                    c("get_job_vectors(index, job_id)"),
                ],
                [
                    c("job_titles"),
                    "job",
                    c('"<job id>"'),
                    c("job_id, company"),
                    c("get_title_vector()") + ", " + c("search_titles()"),
                ],
            ],
            [0.15, 0.17, 0.18, 0.2, 0.3],
        ),
        Spacer(1, 6),
        p(
            c("index_jobs()")
            + " first deletes a job's old chunks (a shorter new description may "
            "have fewer), then adds the new ones and <i>upserts</i> (inserts or replaces) the title. "
            "Because SQLite keeps the full text, "
            + c("rebuild_index()")
            + " can recreate all of "
            "Chroma, for example after changing the model or chunk size."
        ),
        p("2.5 From vectors to a score", H2),
        p(
            c("matching.score_job()")
            + " is pure maths on stored vectors: no database, no model. "
            "It computes three similarities and combines them with the weights from your profile ("
            + c("[weights]")
            + " in profile.toml, defaults title 1.0, interests 1.0, avoid 0.5):"
        ),
        code("""
title    = best similarity between the job title and any of your titles
interest = best similarity between any interest and any chunk
avoid    = best similarity between any avoid item and any chunk

total = w.title * title + w.interests * interest - w.avoid * avoid
"""),
        p(
            'A real example, the current top match (GitLab, "Data Scientist, Enterprise Analytics"): '
            'title 0.82, best interest "predictive modeling" 0.75, avoid 0.47, so '
            "1.0 x 0.82 + 1.0 x 0.75 - 0.5 x 0.47 = <b>1.33</b>. "
            + c("Scores")
            + " keeps every part (and each interest's score by name) so the agent can explain the "
            "number rather than just report it."
        ),
        p(
            c("rank_jobs()")
            + " wraps it all: embed the profile once, loop over saved jobs with "
            'status "new", skip any that fail '
            + c("passes_filters()")
            + " or "
            + c("is_recent()")
            + ", fetch their stored vectors, score, sort, and return the best "
            + c("limit")
            + " as "
            + c("MatchResult(job, scores)")
            + "."
        ),
        p("2.6 So where is the RAG?", H2),
        p(
            "<b>RAG</b> means giving an LLM the relevant documents, retrieved from your own data, "
            "inside its prompt, so its answer is based on them rather than on memory. Both halves are "
            "here:"
        ),
        *bullets(
            [
                "<b>Retrieval</b> is "
                + c("rank_jobs()")
                + " (and "
                + c("search_titles()")
                + ", a classic nearest-neighbour search in Chroma). It narrows thousands of saved "
                "jobs to the few most relevant, using the embeddings.",
                "<b>Augmented generation</b> is "
                + c("review.py")
                + " and the agent. The retrieved "
                "jobs' full text goes into Claude's prompt ("
                + c("build_prompt()")
                + ", or the "
                + c("find_matches")
                + " / "
                + c("get_job_details")
                + " tool results), and Claude "
                "writes its judgment from that text.",
                "<b>Grounding.</b> "
                + c("agent_prompts/system.md")
                + " forbids the agent from "
                "describing any job it didn't get from a tool in this conversation, and asks it to "
                "cite every job's id and URL. That's what keeps answers tied to real postings.",
            ]
        ),
        p("2.7 What embeddings can't do", H2),
        p(
            'Embeddings measure topic similarity. They can\'t reliably tell "Senior, 10+ years" from '
            '"Intern" (same topic), they handle "no travel required" poorly (negation), and they '
            "know nothing about your actual experience. That's the reason for Part 3: the cheap "
            "embedding step narrows the field, and the expensive LLM step judges only the finalists.",
            NOTE,
        ),
        PageBreak(),
    ]


def part3() -> list:
    return [
        p("Part 3. What the LLM does", H1),
        p(
            "Claude, the LLM (large language model), runs in exactly two places: "
            + c("review.py")
            + " and the agent. Everything else is ordinary code that runs offline "
            "and is tested without the network. This split is deliberate: LLM calls cost money, take "
            "seconds, and give slightly different answers each time, so they're kept to the steps "
            "that truly need reading and judgment."
        ),
        table(
            [
                ["Step", "Claude?", "Why or why not"],
                ["discover, fetch, store", "no", "pure data plumbing"],
                [
                    "embeddings, chunking, scoring",
                    "no",
                    "a small local model plus maths; fast, free, repeatable",
                ],
                [
                    "filters (place, salary)",
                    "no",
                    "hard rules you set; code applies them exactly",
                ],
                [
                    "review (rerank the top 20)",
                    "<b>yes</b>",
                    (
                        "needs reading the posting against your resume: "
                        "seniority, real duties, avoid items"
                    ),
                ],
                [
                    "chat agent",
                    "<b>yes</b>",
                    "understands your questions and decides which tools to call",
                ],
            ],
            [0.33, 0.12, 0.55],
        ),
        p("3.1 Review: one structured call", H2),
        figure(
            diagram_review(),
            "Figure 5. review.py: the retrieved shortlist goes in, a "
            "checked list of reviews comes out.",
        ),
        *bullets(
            [
                "<b>Instructions</b> ("
                + c("review_instructions()")
                + "): a short intro, then the "
                + c("screen-posting")
                + " skill loaded from "
                + c("agent_prompts/skills.md")
                + " via "
                + c("prompts.load_skill()")
                + ', then "review every job_id exactly once". '
                "The chat agent uses the same skill, so both judge jobs the same way.",
                "<b>The message</b> ("
                + c("build_prompt()")
                + "): your profile, each resume "
                "(after "
                + c("remove_contact_details()")
                + " blanks emails and phone numbers) and "
                "every job with its "
                + c("job_id")
                + ", each wrapped in tags like "
                + c("<resume>...</resume>")
                + " so Claude sees clear boundaries.",
                "<b>Structured output</b>: "
                + c("client.beta.messages.parse(output_format=ReviewBatch)")
                + " makes Claude's answer match the Pydantic model, and the library checks it. "
                "You get Python objects, never text to pick apart.",
                "<b>Settings</b>: model "
                + c("claude-opus-5-5")
                + " (change with "
                + c("JOBS_SEARCH_REVIEW_MODEL")
                + '), effort "medium", and server-side fallbacks '
                "so a rare safety decline is retried on another model instead of failing.",
                "<b>Safety nets</b>: a refusal, a cut-off answer or a malformed answer raises a clear "
                "error; a job Claude skipped is kept at the bottom with "
                + c("review=None")
                + "; "
                "reviews for ids that were never sent are ignored.",
            ]
        ),
        table(
            [
                ["JobReview field", "Meaning"],
                [
                    c("job_id"),
                    "copied from the prompt, used to match the review back to its job",
                ],
                [c("fit"), "1 (clearly wrong) to 10 (apply today); the main sort key"],
                [
                    c("verdict"),
                    c('"strong"') + ", " + c('"maybe"') + " or " + c('"skip"'),
                ],
                [
                    c("reasons") + ", " + c("concerns"),
                    "short, specific lists, quoting the posting where possible",
                ],
            ],
            [0.3, 0.7],
        ),
        p("3.2 The agent: Claude with tools", H2),
        p(
            "An <b>agent</b> is an LLM that decides for itself which <b>tools</b> (functions) to call. "
            "LangChain's "
            + c("create_agent()")
            + " runs the loop below. Claude never sees the tool "
            "code: only each tool's name, typed arguments and docstring, which is why the docstrings "
            "in " + c("tools.py") + " are written as instructions for Claude."
        ),
        figure(
            diagram_agent_loop(),
            "Figure 6. One chat turn. The tools run your own code; "
            "Claude only chooses which to call and writes the reply.",
        ),
        table(
            [
                ["Tool", "Arguments", "Calls", "Returns (as text)"],
                [
                    c("find_matches"),
                    c("title, new_since_days, limit"),
                    c("matching.rank_jobs()"),
                    "numbered list with score parts, salary, id, url",
                ],
                [
                    c("get_job_details"),
                    c("job_id"),
                    c("store.get_job()") + ", " + c("get_status()"),
                    "full posting + your status",
                ],
                [
                    c("set_status"),
                    c("job_id, status"),
                    c("store.set_status()"),
                    "one-line confirmation, or an error message",
                ],
            ],
            [0.18, 0.27, 0.25, 0.30],
        ),
        Spacer(1, 6),
        *bullets(
            [
                "<b>System prompt</b> ("
                + c("agent.system_prompt()")
                + "): "
                + c("system.md")
                + " (rules) + "
                + c("skills.md")
                + " (step-by-step recipes) + your profile.",
                "<b>Identity is never an argument.</b> The database, index and profile come from a "
                + c("ToolContext")
                + " fixed at start-up, so Claude can't point a tool at other data.",
                "<b>Memory</b>: the model remembers nothing between calls, so "
                + c("ask()")
                + " resends the whole conversation (questions, tool calls, results, answers) each turn.",
                "<b>Errors become text</b>, so the agent can explain or retry instead of crashing.",
            ]
        ),
        p("3.3 Privacy and cost", H2),
        p(
            "Review and chat send text to Anthropic's API: your profile, your resume (contact details "
            "removed) and job postings. Nothing else in the pipeline leaves your computer except "
            "requests to the public job boards and Common Crawl. Your API key is read from "
            + c(".env")
            + " by the Anthropic library; the code never reads or prints it."
        ),
        PageBreak(),
    ]


def module_section(name: str, role: str, rows: list[list[str]]) -> list:
    return [
        KeepTogether([
            p(name, H3),
            p(role),
            table([["Class / function", "What it does and how it serves the goal"], *rows],
                  [0.34, 0.66]),
        ]),
        Spacer(1, 4),
    ]  # fmt: skip


def part4() -> list:
    story = [
        p("Part 4. Module reference", H1),
        p(
            "Every module in "
            + c("src/jobs_search/")
            + ", in pipeline order. Each table lists the "
            "classes and functions you're most likely to read or change."
        ),
    ]
    sections = [
        ("models.py", "The one data shape every module shares.", [
            [c("Source"), "the allowed board names, as a " + c("Literal") + " type"],
            [c("Job(BaseModel)"), "one posting; Pydantic checks every field's type when a Job is made"],
        ]),
        ("discovery.py", "Builds " + c("companies.csv") + " so you never maintain a company list.", [
            [c("Company"), "a board: " + c("source") + " + " + c("slug") + "; frozen so it can go in a set"],
            [c("company_from_url(url)"), "turns " + c("jobs.lever.co/spotify/...") + " into " + c("Company(lever, spotify)")],
            [c("common_crawl_candidates(pages)"), ("asks Common Crawl's public index for board addresses; "
             "returns what it can if the service is busy")],
            [c("seed_candidates(path)"), "reads " + c("company_seeds.txt") + " (starter guesses)"],
            [c("board_exists(company)"), "asks the real API: 200 = real, 404 = not"],
            [c("discover(path, candidates)"), "checks only new candidates, adds the real ones, saves the file, "
             "returns a " + c("DiscoveryReport")],
            [c("load_companies / save_companies / remove_companies"), "read, write (sorted, no duplicates) and prune the CSV"],
        ]),
        ("fetchers/", "One file per board, all following the same pattern: call the API with a "
         "timeout, " + c("raise_for_status()") + ", then a pure " + c("parse_*()") + " function "
         "that is tested on saved JSON.", [
            [c("FETCHERS") + " (in __init__.py)", "dict from board name to fetch function; pipeline.py looks fetchers up here"],
            [c("fetch_greenhouse / fetch_lever / fetch_ashby(slug)"), "download one company's board -> " + c("list[Job]")],
            [c("parse_greenhouse / parse_lever / parse_ashby"), "API JSON -> Job (ids, salary, dates, plain-text description)"],
            [c("helpers.html_to_text(html)"), "HTML -> readable text (BeautifulSoup)"],
            [c("helpers.build_location(places, is_remote)"), "one consistent location string, adding \"Remote\" when needed"],
        ]),
        ("store.py", "SQLite: the source of truth.", [
            [c("connect(path)"), "opens " + c("data/jobs.db") + " and creates the table if needed"],
            [c("content_hash(job)"), "SHA-256 fingerprint of the parts that matter; dates excluded"],
            [c("save_jobs(conn, jobs)"), "insert new jobs, update changed ones (keeping your status), return " + c("SaveResult")],
            [c("get_job / list_jobs(statuses)"), "read jobs back as " + c("Job") + " objects"],
            [c("get_status / set_status"), "your decisions: new, applied, rejected, hidden"],
        ]),
        ("embeddings.py", "The local embedding model.", [
            [c("Embedder"), "the type of any embed function: texts in, vectors out"],
            [c("get_model()"), "loads fastembed's " + c("TextEmbedding") + " once and caches it"],
            [c("embed_texts(texts)"), "the real embedder used by fetch, match, review and chat"],
        ]),
        ("rag.py", "Chunking and the Chroma vector index.", [
            [c("JobIndex"), "holds the Chroma client and the two collections"],
            [c("open_index(path)"), "opens (or creates) " + c("data/chroma")],
            [c("chunk_text / split_long_paragraph"), "description -> pieces of at most 1,000 characters"],
            [c("index_jobs(index, jobs, embed)"), "replace each job's chunk vectors and title vector"],
            [c("get_job_vectors / get_title_vector"), "read stored vectors back for scoring"],
            [c("search_titles(index, query, embed)"), "nearest-neighbour title search -> " + c("list[TitleMatch]")],
            [c("rebuild_index(conn, index, embed)"), "recreate the whole index from SQLite"],
        ]),
        ("profile.py", "Your preferences, checked as soon as they're loaded.", [
            [c("Profile"), "titles, interests, avoid, filters, weights; " + c("StrictModel") + " rejects typos in field names"],
            [c("Interest, AvoidItem, Filters, Weights"), "the pieces of profile.toml"],
            [c("load_profile(folder)"), "reads profile.toml with " + c("tomllib")],
            [c("load_resumes(folder)"), "reads .md / .txt resumes -> " + c("list[Resume]")],
        ]),
        ("matching.py", "Filters and embedding scores.", [
            [c("passes_location / passes_salary / passes_filters"), "your hard rules: Twin Cities or US-remote, salary floor"],
            [c("is_us_remote / mentions_us"), "reads messy location strings like \"Remote; Berlin\""],
            [c("embed_profile(profile, embed, title)"), "profile -> " + c("ProfileVectors") + " (fresh each query)"],
            [c("cosine_similarity / best_similarity"), "the maths, in plain Python"],
            [c("score_job(...)"), "one job's " + c("Scores") + " from stored vectors"],
            [c("rank_jobs(...)"), "filter, score and sort -> " + c("list[MatchResult]")],
        ]),
        ("pipeline.py", "The collect half, joined up.", [
            [c("fetch_and_index(...)"), "for each company: fetch, save, pick and index; returns " + c("FetchReport")],
            [c("jobs_to_index(...)"), "jobs passing filters that are new, changed or missing from the index"],
        ]),
        ("review.py", "Claude reranks the shortlist (see Part 3).", [
            [c("JobReview, ReviewBatch"), "the structured output Claude must return"],
            [c("ReviewedJob"), "a job with both scores: " + c("match_score") + " and " + c("review")],
            [c("remove_contact_details / format_profile / format_job / build_prompt"), "build the one message Claude reads"],
            [c("call_claude(instructions, message)"), "the only Anthropic API call in review"],
            [c("review_jobs(results, profile, resumes, ask)"), "call, match reviews to jobs, sort with " + c("sort_key")],
        ]),
        ("prompts.py", "Loads the agent's instructions from Markdown.", [
            [c("load_prompt(name)"), "reads " + c("agent_prompts/<name>.md")],
            [c("split_skills / load_skill(name)"), "splits skills.md at each " + c("## ") + " heading; one skill by name"],
            [c("agent_system_prompt()"), "system.md + skills.md"],
        ]),
        ("tools.py and agent.py", "The chat agent (see Part 3).", [
            [c("ToolContext"), "database path, index, profile and embedder, fixed at start-up"],
            [c("make_tools(ctx)"), "builds the three " + c("@tool") + " functions as closures over ctx"],
            [c("format_match / format_job_details"), "turn results into text for Claude (and for " + c("match") + ")"],
            [c("chat_model()"), "LangChain's " + c("ChatAnthropic") + " with the model name and effort"],
            [c("build_agent(ctx, model)"), c("create_agent()") + " with tools and system prompt"],
            [c("ask / chat_loop"), "one turn / the terminal conversation"],
        ]),
        ("main.py", "The command line: thin wiring, no logic of its own.", [
            [c("build_parser()"), "argparse commands and options (and the free " + c("--help") + ")"],
            [c("run_discover / run_fetch / run_match / run_review / run_chat"), "open what each command needs and call the modules"],
        ]),
        ("evaluation.py and evals/", "Measures ranking quality against your labels.", [
            [c("pick_jobs_to_label()"), "15 top-ranked + 15 random filter-passing jobs, shuffled"],
            [c("load_labels / write_label_sheet"), "the CSV you fill in with good / bad"],
            [c("evaluate(...)"), "scores labeled jobs ignoring filters -> " + c("EvalReport")],
            [c("precision_at_k / find_good_below_bad"), "the headline number and where ranking goes wrong"],
        ]),
    ]  # fmt: skip
    for name, role, rows in sections:
        story.extend(module_section(name, role, rows))
    story.append(PageBreak())
    return story


def part5() -> list:
    return [
        p("Part 5. Testing and evaluation", H1),
        p("5.1 Tests run offline", H2),
        p(
            "All 170 tests run without the network, an API key or the embedding model. The trick is "
            "<b>dependency injection</b>: anything slow, costly or external is passed in as an "
            "argument, so a test can pass a fake."
        ),
        table(
            [
                ["Real thing", "Passed in as", "Fake used in tests"],
                [
                    "embedding model",
                    c("embed") + " (" + c("Embedder") + ")",
                    "counts keywords, e.g. [data, pipeline, sales, 1.0]",
                ],
                [
                    "Claude in review",
                    c("ask") + " (" + c("Asker") + ")",
                    c("FakeClaude") + " returns a ready-made " + c("ReviewBatch"),
                ],
                [
                    "Claude in the agent",
                    c("model"),
                    c("ScriptedModel") + " replays a tool call, then an answer",
                ],
                [
                    "job-board APIs",
                    c("fetchers") + " dict",
                    "functions returning test jobs, or raising 404 / 500",
                ],
                [
                    "board checks",
                    c("check"),
                    "a function that says which slugs are real",
                ],
                [
                    "your files",
                    "environment variables",
                    c("tmp_path") + " copies of " + c("tests/fixtures/"),
                ],
            ],
            [0.25, 0.3, 0.45],
        ),
        Spacer(1, 6),
        code("""
uv run pytest            # all tests
uv run ruff check .      # style and common mistakes
"""),
        p("5.2 The ranking eval", H2),
        p(
            "Tests check the code does what it says; the <b>eval</b> checks the ranking is actually "
            "<i>good for you</i>. You label 30 jobs good or bad in "
            + c("profile/labeled_jobs.csv")
            + "; "
            + c("evals/rank_eval.py")
            + " ranks just those jobs and reports <b>precision@10</b>: "
            "the share of the top 10 you labeled good. Run it before and after any change to scoring, "
            "chunking, the embedding model or "
            + c("agent_prompts/")
            + ", and keep the change only if "
            "the number holds or rises."
        ),
        code("""
uv run python evals/make_label_sheet.py     # once: writes the sheet
# ... label it ...
uv run python evals/rank_eval.py "baseline" # precision@10, saved to history
"""),
        p(
            "Note that this eval measures the embedding ranking only. Claude's review and the agent "
            "need their own eval later.",
            NOTE,
        ),
        p("Appendix A. Settings", H2),
        table(
            [
                ["Environment variable", "Default", "Controls"],
                [
                    c("JOBS_SEARCH_DATA_DIR"),
                    c("data"),
                    "where jobs.db, chroma/ and models/ live",
                ],
                [
                    c("JOBS_SEARCH_PROFILE_DIR"),
                    c("profile"),
                    "your profile, resumes and labels",
                ],
                [c("JOBS_SEARCH_COMPANIES_FILE"), c("companies.csv"), "the board list"],
                [
                    c("JOBS_SEARCH_PROMPTS_DIR"),
                    c("agent_prompts"),
                    "the agent's instructions",
                ],
                [
                    c("JOBS_SEARCH_EMBEDDING_MODEL"),
                    c("BAAI/bge-small-en-v1.5"),
                    "embedding model (re-index after changing)",
                ],
                [
                    c("JOBS_SEARCH_REVIEW_MODEL"),
                    c("claude-opus-5-5"),
                    "Claude model for review",
                ],
                [
                    c("JOBS_SEARCH_AGENT_MODEL"),
                    c("claude-opus-5-5"),
                    "Claude model for chat",
                ],
                [
                    c("ANTHROPIC_API_KEY") + " (in .env)",
                    "-",
                    "your key; never committed",
                ],
            ],
            [0.37, 0.28, 0.35],
        ),
        p("Appendix B. Glossary", H2),
        table(
            [
                ["Term", "Meaning"],
                ["agent", "an LLM that decides which tools to call to answer you"],
                [
                    "API",
                    "a web address that returns data (here JSON) for programs instead of pages for people",
                ],
                [
                    "chunk",
                    "a piece of a job description, at most 1,000 characters, embedded separately",
                ],
                ["Chroma", "the vector database that stores embeddings"],
                [
                    "cosine similarity",
                    "how closely two vectors point the same way; 1.0 = same meaning",
                ],
                [
                    "embedding / vector",
                    "a list of numbers (384 here) that captures a text's meaning",
                ],
                [
                    "eval",
                    "a repeatable measurement of quality against labeled examples",
                ],
                ["LLM", "large language model; here Claude"],
                ["Pydantic model", "a class that checks its data's types when created"],
                [
                    "RAG",
                    "retrieval-augmented generation: retrieve relevant documents, put them in the LLM's prompt",
                ],
                ["slug", "a company's short URL name on a board, e.g. gitlab"],
                [
                    "structured output",
                    "an LLM answer forced to match a schema, returned as objects",
                ],
                [
                    "system prompt",
                    "instructions sent to the LLM before the conversation",
                ],
                [
                    "tool",
                    "a function the agent may call; it sees only the name, arguments and docstring",
                ],
            ],
            [0.25, 0.75],
        ),
    ]


def number_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(inch, 0.6 * inch, "jobs_search tutorial")
        canvas.drawRightString(letter[0] - inch, 0.6 * inch, f"page {doc.page}")
    canvas.restoreState()


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=inch, rightMargin=inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="jobs_search: how the job-search agent works", author="jobs_search",
    )  # fmt: skip
    story = title_page() + part1() + part2() + part3() + part4() + part5()
    doc.build(story, onFirstPage=number_pages, onLaterPages=number_pages)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

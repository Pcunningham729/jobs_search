"""Build docs/walkthrough.pdf: hands-on walkthroughs, using YOUR real data.

Run from the repo root (after `uv run jobs-search fetch` has saved some jobs):

    uv run --with reportlab python docs/build_walkthrough.py

Unlike the tutorial, this document is built from your own database and
profile, so every number and job in it is real. That also makes it personal,
so docs/walkthrough.pdf is gitignored. This script holds no personal data.

It never calls Claude and never changes your database: it only reads.
"""

import sys
import textwrap
from datetime import date
from pathlib import Path

from reportlab.graphics.shapes import Drawing
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, PageBreak, SimpleDocTemplate, Spacer

# Reuse the styles and drawing helpers from the tutorial builder (same folder).
sys.path.insert(0, str(Path(__file__).parent))
from build_tutorial import (
    COLLECT,
    DATA,
    H1,
    H2,
    H3,
    LLM,
    MUTED,
    NOTE,
    PAGE_WIDTH,
    SMALL,
    USE,
    arrow,
    box,
    bullets,
    c,
    figure,
    lane_label,
    note,
    p,
    table,
)
from build_tutorial import code as unwrapped_code

from jobs_search.discovery import companies_path, load_companies
from jobs_search.embeddings import embed_texts
from jobs_search.matching import (
    cosine_similarity,
    embed_profile,
    is_recent,
    passes_filters,
    rank_jobs,
)
from jobs_search.profile import load_profile, profile_dir
from jobs_search.rag import (
    chroma_path,
    chunk_text,
    get_job_vectors,
    get_title_vector,
    open_index,
)
from jobs_search.review import review_instructions
from jobs_search.store import connect, database_path, list_jobs
from jobs_search.tools import ToolContext, format_match, make_tools

OUTPUT = Path(__file__).parent / "walkthrough.pdf"


# ---------------------------------------------------------------------------
# Gather real facts from the database (read-only)
# ---------------------------------------------------------------------------


def gather() -> dict:
    """Everything the document quotes, computed from your data right now."""
    conn = connect(database_path())
    index = open_index(chroma_path())
    profile = load_profile(profile_dir())

    every_job = list_jobs(conn)
    new_jobs = list_jobs(conn, statuses=["new"])
    passing = [job for job in new_jobs if passes_filters(job, profile.filters)]
    indexed = [job for job in passing if get_title_vector(index, job.id) is not None]
    recent = [job for job in indexed if is_recent(job, 7)]

    ranked = rank_jobs(conn, index, profile, embed_texts, limit=len(indexed) or 1)
    top, bottom = ranked[0], ranked[-1]

    # Trace the top job in detail.
    row = conn.execute(
        "SELECT id, source, company, title, status, first_seen_at, content_hash "
        "FROM jobs WHERE id = ?",
        (top.job.id,),
    ).fetchone()
    chunks = chunk_text(top.job.description)
    profile_vectors = embed_profile(profile, embed_texts)
    title_vector = get_title_vector(index, top.job.id)
    title_scores = {
        title.label: cosine_similarity(title.vector, title_vector)
        for title in profile_vectors.titles
    }

    # Real tool output, exactly what the agent would get back. (Read-only
    # tools only: set_status is never called here.)
    ctx = ToolContext(
        db_path=database_path(), index=index, profile=profile, embed=embed_texts
    )
    tools = {tool.name: tool for tool in make_tools(ctx)}
    first_title = profile.titles[min(5, len(profile.titles) - 1)]
    tool_find = tools["find_matches"].invoke({"new_since_days": 7, "limit": 3})
    # The chat example asks about "the first one" in this week's list, so
    # follow that job (or the overall top job if nothing is that recent).
    this_week = rank_jobs(conn, index, profile, embed_texts, new_since_days=7, limit=1)
    chat_job = this_week[0].job if this_week else top.job
    tool_details = tools["get_job_details"].invoke({"job_id": chat_job.id})
    by_title = rank_jobs(conn, index, profile, embed_texts, title=first_title, limit=3)
    conn.close()

    return {
        "profile": profile,
        "companies": len(load_companies(companies_path())),
        "counts": {
            "all": len(every_job),
            "new": len(new_jobs),
            "passing": len(passing),
            "indexed": len(indexed),
            "recent": len(recent),
            "ranked": len(ranked),
        },
        "ranked": ranked,
        "top": top,
        "bottom": bottom,
        "row": dict(row),
        "chunks": chunks,
        "chunk_vectors": len(get_job_vectors(index, top.job.id)),
        "title_scores": title_scores,
        "tool_find": tool_find,
        "tool_details": tool_details,
        "chat_job": chat_job,
        "first_title": first_title,
        "by_title": by_title,
    }


def shorten(text: str, max_lines: int, max_chars: int = 95) -> str:
    """The first few lines of `text`, each cut to fit the page."""
    lines = []
    for line in text.splitlines()[:max_lines]:
        lines.append(line if len(line) <= max_chars else line[: max_chars - 3] + "...")
    if len(text.splitlines()) > max_lines:
        lines.append("    ... (rest cut for this document)")
    return "\n".join(lines)


def code(text: str):
    """A code block whose long lines wrap (indented) instead of running off the page."""
    lines = []
    for line in text.strip("\n").splitlines():
        indent = " " * (len(line) - len(line.lstrip()) + 4)
        lines.extend(textwrap.wrap(line, 88, subsequent_indent=indent) or [""])
    return unwrapped_code("\n".join(lines))


def esc(text: str) -> str:
    """Make text safe inside a Paragraph (which reads < > & as markup)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------


def diagram_three_ways() -> Drawing:
    """Which modules each command uses: agent.py only appears in chat."""
    w, h = PAGE_WIDTH, 300
    d = Drawing(w, h)
    col_w = (w - 2 * 18) / 3
    xs = [0, col_w + 18, 2 * (col_w + 18)]
    titles = [
        "jobs-search match",
        "jobs-search review",
        "jobs-search chat",
    ]
    for x, title in zip(xs, titles):
        lane_label(d, x, 286, title)

    bh = 34
    # match
    box(d, xs[0], 236, col_w, bh, ["main.py", "reads your options"], USE)
    box(
        d,
        xs[0],
        176,
        col_w,
        bh,
        ["matching.rank_jobs()", "retrieval: filter + score"],
        USE,
    )
    box(d, xs[0], 116, col_w, bh, ["printed list", "best first"], USE)
    arrow(d, xs[0] + col_w / 2, 236, xs[0] + col_w / 2, 176 + bh)
    arrow(d, xs[0] + col_w / 2, 176, xs[0] + col_w / 2, 116 + bh)

    # review
    box(d, xs[1], 236, col_w, bh, ["main.py", "fixed steps"], USE)
    box(
        d,
        xs[1],
        176,
        col_w,
        bh,
        ["matching.rank_jobs()", "same retrieval: top 20"],
        USE,
    )
    box(d, xs[1], 116, col_w, bh, ["review.py", "ONE call to Claude"], LLM)
    box(d, xs[1], 56, col_w, bh, ["printed list", "sorted by Claude's fit"], USE)
    arrow(d, xs[1] + col_w / 2, 236, xs[1] + col_w / 2, 176 + bh)
    arrow(d, xs[1] + col_w / 2, 176, xs[1] + col_w / 2, 116 + bh)
    arrow(d, xs[1] + col_w / 2, 116, xs[1] + col_w / 2, 56 + bh)

    # chat
    box(d, xs[2], 236, col_w, bh, ["you type a message", "any time, any wording"], USE)
    box(
        d,
        xs[2],
        176,
        col_w,
        bh,
        ["agent.py: Claude decides", "answer now, or use a tool?"],
        LLM,
    )
    box(d, xs[2], 116, col_w, bh, ["tools.py", "find_matches / details / status"], USE)
    box(d, xs[2], 56, col_w, bh, ["matching.rank_jobs()", "store.py (same code)"], USE)
    arrow(d, xs[2] + col_w / 2, 236, xs[2] + col_w / 2, 176 + bh)
    arrow(d, xs[2] + col_w * 0.35, 176, xs[2] + col_w * 0.35, 116 + bh)
    arrow(d, xs[2] + col_w * 0.65, 116 + bh, xs[2] + col_w * 0.65, 176)
    arrow(d, xs[2] + col_w / 2, 116, xs[2] + col_w / 2, 56 + bh)
    note(d, xs[2] + col_w / 2, 42, "loops until Claude answers")

    box(d, 0, 2, w, 26, [("All three read the same two databases: data/jobs.db (job text, "
        "your statuses) and data/chroma (stored vectors)")], DATA, bold_first=False)  # fmt: skip
    return d


def diagram_funnel(counts: dict) -> Drawing:
    """How the jobs in your database narrow down to the list you see."""
    steps = [
        (
            f"{counts['all']} jobs saved in SQLite",
            "everything fetch has ever saved",
            DATA,
        ),
        (f'{counts["new"]} with status "new"', "not applied / rejected / hidden", DATA),
        (
            f"{counts['passing']} pass your filters",
            "Twin Cities or US-remote, salary floor",
            USE,
        ),
        (f"{counts['indexed']} have stored vectors", "were embedded during fetch", USE),
        (f"{counts['ranked']} scored and sorted", "pure maths on stored vectors", USE),
        ("top 10 printed", "or top 20 sent to Claude by review", COLLECT),
    ]
    w, h = PAGE_WIDTH, 40 * len(steps) + 10
    d = Drawing(w, h)
    for number, (label, detail, fill) in enumerate(steps):
        width = w * (1 - number * 0.1)
        x = (w - width) / 2
        y = h - (number + 1) * 40
        box(d, x, y + 4, width, 30, [label, detail], fill)
    return d


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def build(facts: dict) -> list:
    counts = facts["counts"]
    top = facts["top"]
    bottom = facts["bottom"]
    profile = facts["profile"]
    scores = top.scores
    weights = profile.weights
    best_title = max(facts["title_scores"], key=facts["title_scores"].get)
    chat_job = facts["chat_job"]

    story = [
        p("Using jobs_search: walkthroughs", H1),
        p(
            f"Built on {date.today().isoformat()} from <b>your own database and profile</b>, so "
            "every job, count and score below is real. Rebuild it any time with "
            + c("uv run --with reportlab python docs/build_walkthrough.py")
            + ". It's "
            "gitignored because it contains your data."
        ),
        p("The one idea to hold onto", H2),
        p("The app has two halves that run at different times:"),
        *bullets(
            [
                "<b>Building the database</b> ("
                + c("discover")
                + ", "
                + c("fetch")
                + "): slow, "
                "run every day or two. It saves jobs to SQLite and their vectors to Chroma. "
                "It doesn't find you anything by itself; it just stocks the shelves.",
                "<b>Asking the database questions</b> ("
                + c("match")
                + ", "
                + c("review")
                + ", "
                + c("chat")
                + "): fast, run whenever you like. Every one of them starts the same "
                "way: <b>retrieval</b>, meaning "
                + c("matching.rank_jobs()")
                + " picks the "
                "jobs that best fit a query out of everything saved. They differ only in what "
                "happens after that.",
            ]
        ),
        figure(
            diagram_three_ways(),
            "Figure 1. The three ways to ask. agent.py is used ONLY by chat; review.py ONLY "
            "by review; match uses no LLM at all. All three use the same retrieval.",
        ),
        p(
            "<b>A common mix-up:</b> "
            + c("agent.py")
            + " is not part of the fixed pipeline, "
            "and it isn't used by "
            + c("review")
            + ". The fixed path is fetch, then match "
            "(or review). "
            + c("agent.py")
            + " only runs while you're chatting, and all it does "
            "is let Claude call the same functions you can run yourself, in response to what "
            "you type.",
            NOTE,
        ),
        PageBreak(),
        # ---------------- Walkthrough 1 ----------------
        p("Walkthrough 1. First-time setup", H1),
        p("Run every command from the repo folder. You only do this once."),
        code("""
uv sync                                  # install everything in pyproject.toml
uv run jobs-search --help                # lists all seven commands
"""),
        p(
            "For "
            + c("review")
            + " and "
            + c("chat")
            + " (the commands that use Claude), create a "
            "file called " + c(".env") + " in the repo folder containing one line:"
        ),
        code("ANTHROPIC_API_KEY=<your key>"),
        p(
            "Then check your profile loads (it prints your titles; an error names the bad field):"
        ),
        code(
            'uv run python -c "from jobs_search.profile import load_profile, profile_dir; '
            'print(load_profile(profile_dir()).titles)"'
        ),
        Spacer(1, 6),
        # ---------------- Walkthrough 2 ----------------
        p("Walkthrough 2. Your routine", H1),
        p("Step 1. Build or refresh the database", H2),
        code("""
uv run jobs-search discover      # now and then: finds more job boards
uv run jobs-search fetch         # every day or two: new and changed jobs
"""),
        p(
            "Your "
            + c("companies.csv")
            + f" lists <b>{facts['companies']}</b> boards right now. "
            "The first full fetch is slow (every job that passes your filters is embedded); "
            "later fetches are quick because unchanged jobs are skipped. Each board prints one "
            "line, e.g.:"
        ),
        code("[12/73] greenhouse:gitlab 246 jobs, 3 new, 1 changed, 2 indexed"),
        p("Step 2. See your best matches (free, instant)", H2),
        code("uv run jobs-search match"),
        p("This is real output from your database right now (top 3):"),
        code(
            "\n\n".join(
                format_match(n, r) for n, r in enumerate(facts["ranked"][:3], 1)
            )
        ),
        p(
            "How to read it: <b>fit score</b> is the total. The parts in brackets show "
            "<i>why</i>: how close the job title is to one of yours, which of your interests the "
            "description matched best and how strongly, and how close it came to your avoid "
            "list (higher avoid = worse). Walkthrough 3 shows exactly how these are computed."
        ),
        p("Step 3. Let Claude judge the shortlist (a few cents)", H2),
        code("uv run jobs-search review"),
        p(
            "Claude reads the same top 20 next to your resume and gives each a fit from 1 to 10 "
            "with reasons and concerns, then the list is re-sorted by that. Use it when the match "
            "list looks plausible but you can't tell which are realistic for you (walkthrough 5)."
        ),
        p("Step 4. Talk it through (a little per message)", H2),
        code("uv run jobs-search chat"),
        p("Ask in plain English, follow up, and record what you did (walkthrough 6)."),
        # ---------------- Walkthrough 3 ----------------
        p('Walkthrough 3. What happens when you type "match"', H1),
        p(
            "This is <b>retrieval</b>: finding the most relevant jobs among everything saved. "
            "It's what turns the database into a shortlist. No LLM is involved; it's filters "
            "plus arithmetic on vectors that were stored during fetch."
        ),
        p("3.1 The funnel, with your numbers", H2),
        figure(
            diagram_funnel(counts),
            "Figure 2. rank_jobs() narrows your saved jobs down to a ranked list.",
        ),
        p(
            f"Most of the narrowing is your <b>filters</b> ({counts['all']} saved down to "
            f"{counts['passing']}), which are plain rules in "
            + c("matching.passes_filters()")
            + f". The embeddings then <b>order</b> those {counts['ranked']} jobs. Retrieval "
            "here means scoring every job that survives the filters, not just looking up a few "
            "nearest neighbours, because there are few enough to score them all."
        ),
        p("3.2 Following your #1 job through the whole system", H2),
        p(
            f"Your top job is <b>{esc(top.job.title)}</b> at <b>{esc(top.job.company)}</b>. "
            "Here's every place it lives and every number that put it first."
        ),
        p("A. In SQLite (saved by fetch)", H3),
        table(
            [
                ["Column", "Value"],
                *[[c(key), esc(str(value))] for key, value in facts["row"].items()],
                [
                    c("data"),
                    (
                        f"the whole Job as JSON, including the {len(top.job.description):,}-character "
                        "description"
                    ),
                ],
            ],
            [0.25, 0.75],
        ),
        p("B. In Chroma (embedded by fetch)", H3),
        p(
            f"The description was split into <b>{len(facts['chunks'])} chunks</b> and each became "
            f"a vector of 384 numbers ({facts['chunk_vectors']} stored in "
            + c("job_chunks")
            + "). The title became one more vector in "
            + c("job_titles")
            + ". The first few "
            "chunks begin:"
        ),
        code(
            "\n".join(
                f"chunk {n}: {chunk[:80].replace(chr(10), ' ')}..."
                for n, chunk in enumerate(facts["chunks"][:4])
            )
        ),
        p("C. Your profile, embedded fresh for this search", H3),
        p(
            f"Your {len(profile.titles)} titles, {len(profile.interests)} interests and "
            f"{len(profile.avoid)} avoid item(s) are turned into vectors now (this takes a "
            "moment, so editing profile.toml takes effect on the next search)."
        ),
        p("D. The comparisons", H3),
        p(
            "<b>Title:</b> the job title's vector against each of your titles. The best one wins:"
        ),
        table(
            [["Your title", "Similarity to the job title"]]
            + [
                [("<b>" + esc(t) + "</b>") if t == best_title else esc(t), f"{s:.2f}"]
                for t, s in sorted(facts["title_scores"].items(), key=lambda i: -i[1])
            ],
            [0.6, 0.4],
        ),
        Spacer(1, 6),
        p(
            f"<b>Interests:</b> each interest against all {len(facts['chunks'])} chunks, keeping its "
            "best chunk. The best interest overall wins:"
        ),
        table(
            [["Your interest", "Best chunk similarity"]]
            + [
                [
                    ("<b>" + esc(n) + "</b>") if n == scores.best_interest else esc(n),
                    f"{s:.2f}",
                ]
                for n, s in sorted(scores.interests.items(), key=lambda i: -i[1])
            ],
            [0.6, 0.4],
        ),
        Spacer(1, 6),
        p(
            f"<b>Avoid:</b> your avoid item(s) against every chunk; the closest was "
            f"<b>{scores.avoid:.2f}</b>."
        ),
        KeepTogether(
            [
                p("E. The total", H3),
                code(f"""
total = {weights.title} x title + {weights.interests} x interest - {weights.avoid} x avoid
      = {weights.title} x {scores.title:.2f} + {weights.interests} x {scores.interest:.2f} - {weights.avoid} x {scores.avoid:.2f}
      = {scores.total:.2f}
"""),
            ]
        ),
        p(
            f"For contrast, your lowest-ranked passing job is <b>{esc(bottom.job.title)}</b> at "
            f"{esc(bottom.job.company)}: title {bottom.scores.title:.2f}, best interest "
            f"{bottom.scores.interest:.2f}, total <b>{bottom.scores.total:.2f}</b>. Same "
            "arithmetic, smaller numbers."
        ),
        p(
            f"<b>What this tells you:</b> the gap between best ({scores.total:.2f}) and worst "
            f"({bottom.scores.total:.2f}) is real but modest. Embeddings are good at sorting "
            "data jobs above sales jobs; they're bad at seniority, required years, or whether "
            "<i>you</i> could get the job. That's the gap the LLM steps fill.",
            NOTE,
        ),
        PageBreak(),
        # ---------------- Walkthrough 4 ----------------
        p("Walkthrough 4. Asking different questions with match", H1),
        p(
            "Each option changes the <i>query</i> or the <i>filters</i>; the stored job vectors "
            "never change."
        ),
        table(
            [
                ["Command", "What changes"],
                [
                    c("uv run jobs-search match"),
                    "your profile's titles and interests (the default)",
                ],
                [
                    c(f'uv run jobs-search match --title "{facts["first_title"]}"'),
                    "only the title part: that one title replaces your list of titles",
                ],
                [
                    c("uv run jobs-search match --days 7"),
                    f"also drop jobs posted more than 7 days ago ({counts['recent']} of yours qualify)",
                ],
                [c("uv run jobs-search match --limit 25"), "show more results"],
                [
                    c(
                        'uv run jobs-search match --title "data analyst" --days 14 --limit 5'
                    ),
                    "options combine",
                ],
            ],
            [0.55, 0.45],
        ),
        Spacer(1, 6),
        p("Real top 3 for " + c(f'--title "{facts["first_title"]}"') + ":"),
        code(
            "\n".join(
                f"{n}. {r.job.title} at {r.job.company}  (title {r.scores.title:.2f}, total {r.scores.total:.2f})"
                for n, r in enumerate(facts["by_title"], 1)
            )
        ),
        p(
            "To change what <i>every</i> search looks for, edit "
            + c("profile/profile.toml")
            + " instead: titles, interests (one short paragraph each works better than a single "
            "word), the avoid list, filters and weights. No re-fetch is needed."
        ),
        # ---------------- Walkthrough 5 ----------------
        p('Walkthrough 5. What happens when you type "review"', H1),
        p(
            "Review is the fixed pipeline plus <b>one</b> Claude call. Claude doesn't search, "
            "doesn't call tools and doesn't see the database; it only sees what review.py "
            "hands it:"
        ),
        *bullets(
            [
                "Retrieval first: "
                + c("rank_jobs(limit=20)")
                + ", exactly as in walkthrough 3.",
                c("build_prompt()")
                + " writes one message: your profile, your resume(s) with email "
                "and phone removed, then all 20 jobs in full, each with its "
                + c("job_id")
                + ".",
                "Claude gets instructions that start like this (the middle part is the "
                + c("screen-posting")
                + " skill from "
                + c("agent_prompts/skills.md")
                + "):",
            ]
        ),
        code(shorten(review_instructions(), 9)),
        p(
            "Claude must answer in the "
            + c("ReviewBatch")
            + " format (structured output): for "
            "each job_id a fit from 1 to 10, a verdict, reasons and concerns. The code then "
            "re-sorts by fit and prints something like:"
        ),
        code("""
 8/10 strong Data Scientist, Enterprise Analytics (gitlab)
        + Duties center on predictive models and experimentation, matching your interests
        - Asks for 5+ years in SaaS analytics
        https://job-boards.greenhouse.io/gitlab/jobs/...
"""),
        p(
            "(Illustrative output: Claude's real wording and numbers will differ.)",
            SMALL,
        ),
        p(
            "<b>Use review when</b> the match list looks right in topic but you need to know which "
            "jobs are realistic for <i>you</i>. It's one call per run, a few cents.",
            NOTE,
        ),
        PageBreak(),
        # ---------------- Walkthrough 6 ----------------
        p("Walkthrough 6. Chatting with the agent", H1),
        p(
            "Chat is the only part that responds to <i>you</i> in real time. Start it and type:"
        ),
        code("""
uv run jobs-search chat
Job agent (claude-opus-5-5). Type 'quit' to stop.

you> anything new this week?
"""),
        p("What agent.py does with each message", H2),
        *bullets(
            [
                "Sends Claude the system prompt ("
                + c("agent_prompts/system.md")
                + " + "
                + c("skills.md")
                + " + your profile), the whole conversation so far, and your new "
                "message.",
                "Claude replies with either an answer or a <b>tool call</b>, e.g. "
                + c("find_matches(new_since_days=7)")
                + ". It picked that tool and those "
                "arguments itself, from your wording and the tool descriptions.",
                "LangChain runs the tool (real Python: the same "
                + c("rank_jobs()")
                + " as walkthrough 3), sends the text result back to Claude, and repeats until "
                "Claude writes a plain answer.",
            ]
        ),
        p("An example session", H2),
        p(
            "The <b>tool results below are real</b>, produced by running the tools on your "
            "database. Claude's replies are described rather than quoted, since its wording "
            "varies.",
            SMALL,
        ),
        p('You: <i>"anything new this week?"</i>', H3),
        p(
            "Claude calls "
            + c("find_matches(new_since_days=7, limit=3)")
            + " and gets back:"
        ),
        code(facts["tool_find"]),
        p(
            "Claude then answers with a short ranked list: title, company, one line on why "
            "each fits (from the score parts, e.g. the matched interest), and each id and URL, "
            "following the find-new-jobs skill."
        ),
        p(
            "If some of these look wrong for you (a different kind of role, or the wrong "
            "seniority), that's the embedding limit from walkthrough 3: retrieval matched the "
            "topic words. A follow-up question, or running review, is how you catch it.",
            NOTE,
        ),
        p(
            'You: <i>"tell me more about the first one. Is it too senior for me?"</i>',
            H3,
        ),
        p(
            "Claude calls "
            + c(f'get_job_details(job_id="{chat_job.id}")')
            + " and gets the "
            "full posting (first lines shown):"
        ),
        code(shorten(facts["tool_details"], 14)),
        p(
            "Claude reads the requirements and answers using the explain-a-match skill: which of "
            "your interests the duties really involve (quoting the posting), anything on your "
            "avoid list, and a verdict: apply, maybe or skip. In chat it knows your profile but "
            "not your resume, so for experience-level questions " + c("review") + " is "
            "better."
        ),
        p('You: <i>"I applied to it."</i>', H3),
        p(
            "Claude calls "
            + c(f'set_status(job_id="{chat_job.id}", status="applied")')
            + ", "
            "which returns:"
        ),
        code(
            f"Marked {chat_job.title} at {chat_job.company} ({chat_job.id}) as applied."
        ),
        p(
            "That writes to SQLite. From now on "
            + c("rank_jobs()")
            + " skips this job, so it "
            "won't appear in " + c("match") + ", " + c("review") + " or the next chat. "
            "(This document didn't actually change your data.)"
        ),
        p("Other things to try", H2),
        *bullets(
            [
                '"Show me remote research analyst jobs."',
                '"Why did you rank the second one above the third?"',
                '"Does any of these involve a lot of travel?"',
                '"Hide the sales ones."',
                "Type "
                + c("quit")
                + " to stop. The conversation isn't saved between sessions.",
            ]
        ),
        PageBreak(),
        # ---------------- Walkthrough 7 ----------------
        p("Walkthrough 7. Checking the ranking is any good", H1),
        code("""
uv run jobs-search label            # once: writes profile/labeled_jobs.csv (30 jobs)
# open the CSV, write good or bad in the label column, save
uv run jobs-search eval "baseline"  # precision@10 + where ranking goes wrong
"""),
        p(
            "Run "
            + c("eval")
            + " again after any change to your profile weights, the avoid "
            "list, chunking or the model, with a note like "
            + c('"rewrote avoid item"')
            + ". It prints the change since last time and warns if it went down."
        ),
        # ---------------- Cheat sheet ----------------
        p("Cheat sheet", H1),
        table(
            [
                ["I want to...", "Run"],
                ["find more companies", c("uv run jobs-search discover")],
                ["get the latest jobs", c("uv run jobs-search fetch")],
                [
                    "try fetch quickly on a few boards",
                    c("uv run jobs-search fetch --max-companies 3"),
                ],
                ["see my best matches", c("uv run jobs-search match")],
                [
                    "search a different title",
                    c('uv run jobs-search match --title "research analyst"'),
                ],
                ["see only this week's jobs", c("uv run jobs-search match --days 7")],
                ["know which are realistic for me", c("uv run jobs-search review")],
                ["ask questions / mark jobs applied", c("uv run jobs-search chat")],
                [
                    "measure ranking quality",
                    c("uv run jobs-search label")
                    + ", then "
                    + c("uv run jobs-search eval"),
                ],
                ["change what searches look for", "edit " + c("profile/profile.toml")],
                ["see every option", c("uv run jobs-search <command> --help")],
            ],
            [0.45, 0.55],
        ),
        p("When something goes wrong", H2),
        table(
            [
                ["You see", "It means"],
                [
                    '"No matches. Have you run fetch?"',
                    "the database is empty, or every job failed your filters",
                ],
                [
                    '"Common Crawl isn\'t answering"',
                    "a free service is busy; discover still checks the seed list",
                ],
                [
                    "a 401 / authentication error in review or chat",
                    c(".env") + " is missing or the key is wrong",
                ],
                [
                    '"Need at least 10 labeled jobs"',
                    "label more rows in " + c("profile/labeled_jobs.csv"),
                ],
                [
                    "PermissionError writing a PDF",
                    "the PDF is open in a viewer; close it and rerun",
                ],
            ],
            [0.45, 0.55],
        ),
    ]
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        inch, 0.6 * inch, "jobs_search walkthroughs (personal: built from your data)"
    )
    canvas.drawRightString(letter[0] - inch, 0.6 * inch, f"page {doc.page}")
    canvas.restoreState()


def main() -> None:
    print("Reading your database and profile (this loads the embedding model)...")
    facts = gather()
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=inch, rightMargin=inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="jobs_search walkthroughs", author="jobs_search",
    )  # fmt: skip
    doc.build(build(facts), onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

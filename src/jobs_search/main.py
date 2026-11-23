"""The command line: one entry point for every step of the pipeline.

    uv run jobs-search discover   find companies with job boards (companies.csv)
    uv run jobs-search fetch      download, save and index their jobs
    uv run jobs-search match      your best matches (embeddings only, free)
    uv run jobs-search review     Claude reranks the best matches (costs a little)
    uv run jobs-search chat       talk to the job agent (costs a little)

Add --help to any of them to see its options, e.g. `uv run jobs-search match --help`.

`jobs-search` is set up in pyproject.toml under [project.scripts]: it runs
main() below. Like the tools, this file stays thin: it reads the options,
calls functions from the other modules and prints their results.
"""

import argparse

from jobs_search.discovery import (
    common_crawl_candidates,
    companies_path,
    discover,
    load_companies,
    seed_candidates,
)
from jobs_search.embeddings import embed_texts
from jobs_search.matching import rank_jobs
from jobs_search.pipeline import fetch_and_index
from jobs_search.profile import load_profile, load_resumes, profile_dir
from jobs_search.rag import chroma_path, open_index
from jobs_search.store import connect, database_path
from jobs_search.tools import format_match

SEEDS_FILE = "company_seeds.txt"


# ---------------------------------------------------------------------------
# One function per command. Each gets `args`: the options typed after it.
# ---------------------------------------------------------------------------


def run_discover(args: argparse.Namespace) -> None:
    path = companies_path()
    print("Collecting candidate companies...")
    candidates = seed_candidates(path.parent / SEEDS_FILE)
    print(f"  {len(candidates)} from {SEEDS_FILE}")
    if not args.skip_crawl:
        from_crawl = common_crawl_candidates(pages=args.crawl_pages)
        print(f"  {len(from_crawl)} from Common Crawl")
        candidates |= from_crawl

    print("Checking new candidates against the job-board APIs...")
    report = discover(path, candidates)
    print(
        f"\nAdded {len(report.added)} boards. Already known: {report.already_known}. "
        f"Not real boards: {report.not_found}. Couldn't check: {report.errors}."
    )
    print(f"{path} now lists {len(load_companies(path))} boards.")


def run_fetch(args: argparse.Namespace) -> None:
    path = companies_path()
    companies = load_companies(path)
    if not companies:
        print(f"No companies in {path}. Run: uv run jobs-search discover")
        return
    if args.max_companies:
        companies = companies[: args.max_companies]

    conn = connect(database_path())
    report = fetch_and_index(
        conn,
        open_index(chroma_path()),
        companies,
        load_profile(profile_dir()),
        embed_texts,
        companies_file=path,
    )
    conn.close()
    print(
        f"\nFetched {report.companies_fetched} boards: {report.jobs_seen} jobs, "
        f"{report.new_jobs} new, {report.changed_jobs} changed, "
        f"{report.indexed_jobs} indexed."
    )
    if report.failed:
        print("Failed (will retry next time):", ", ".join(report.failed))
    if report.removed:
        print("Removed closed boards:", ", ".join(report.removed))


def run_match(args: argparse.Namespace) -> None:
    conn = connect(database_path())
    results = rank_jobs(
        conn,
        open_index(chroma_path()),
        load_profile(profile_dir()),
        embed_texts,
        title=args.title,
        new_since_days=args.days,
        limit=args.limit,
    )
    conn.close()
    if not results:
        print("No matches. Have you run `uv run jobs-search fetch`?")
    for position, result in enumerate(results, start=1):
        print(format_match(position, result) + "\n")


def run_review(args: argparse.Namespace) -> None:
    # Imported here so the free commands never load the Claude libraries.
    from jobs_search.review import review_jobs, review_model_name

    profile = load_profile(profile_dir())
    conn = connect(database_path())
    shortlist = rank_jobs(
        conn, open_index(chroma_path()), profile, embed_texts, limit=args.limit
    )
    conn.close()
    if not shortlist:
        print("No matches to review. Have you run `uv run jobs-search fetch`?")
        return

    print(f"Asking {review_model_name()} to review {len(shortlist)} jobs...")
    resumes = load_resumes(profile_dir() / "resumes")
    for item in review_jobs(shortlist, profile, resumes):
        job = item.job
        if item.review is None:
            print(f"\n  ?   {job.title} ({job.company}): not reviewed")
            continue
        review = item.review
        print(f"\n{review.fit:2}/10 {review.verdict:6} {job.title} ({job.company})")
        for reason in review.reasons:
            print(f"        + {reason}")
        for concern in review.concerns:
            print(f"        - {concern}")
        print(f"        {job.url}")


def run_chat(args: argparse.Namespace) -> None:
    from jobs_search.agent import build_agent, chat_loop
    from jobs_search.tools import ToolContext

    ctx = ToolContext(
        db_path=database_path(),
        index=open_index(chroma_path()),
        profile=load_profile(profile_dir()),
        embed=embed_texts,
    )
    chat_loop(build_agent(ctx))


# ---------------------------------------------------------------------------
# Reading the command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Describe every command and option. argparse (built into Python) then
    reads what you typed, checks it, and writes the --help text for free."""
    parser = argparse.ArgumentParser(
        prog="jobs-search", description="Find jobs worth applying to."
    )
    # "Subparsers" are the commands: jobs-search <command> [options].
    commands = parser.add_subparsers(dest="command", required=True)

    discover_cmd = commands.add_parser(
        "discover", help="find companies with job boards"
    )
    discover_cmd.add_argument(
        "--crawl-pages", type=int, default=1,
        help="pages of Common Crawl results per board (more = more companies, slower)",
    )  # fmt: skip
    discover_cmd.add_argument(
        "--skip-crawl", action="store_true", help="only check company_seeds.txt"
    )
    discover_cmd.set_defaults(run=run_discover)  # which function runs this command

    fetch_cmd = commands.add_parser("fetch", help="download, save and index jobs")
    fetch_cmd.add_argument(
        "--max-companies",
        type=int,
        help="only fetch the first N boards (for a quick try)",
    )
    fetch_cmd.set_defaults(run=run_fetch)

    match_cmd = commands.add_parser("match", help="show your best matches")
    match_cmd.add_argument(
        "--title", help='search this title instead, e.g. "data engineer"'
    )
    match_cmd.add_argument(
        "--days", type=int, help="only jobs posted in the last N days"
    )
    match_cmd.add_argument("--limit", type=int, default=10, help="how many to show")
    match_cmd.set_defaults(run=run_match)

    review_cmd = commands.add_parser("review", help="Claude reranks your best matches")
    review_cmd.add_argument("--limit", type=int, default=20, help="how many to review")
    review_cmd.set_defaults(run=run_review)

    chat_cmd = commands.add_parser("chat", help="talk to the job agent")
    chat_cmd.set_defaults(run=run_chat)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Read the command line and run the chosen command.

    `argv` is only passed by tests; normally argparse reads what you typed.
    """
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

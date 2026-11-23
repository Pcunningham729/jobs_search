"""Make a sheet of jobs for you to label good or bad.

Run from the repo root:

    uv run python evals/make_label_sheet.py

It writes profile/labeled_jobs.csv (it won't overwrite an existing one).
Open it in Excel or any editor, and in the `label` column write `good` or
`bad` for each job. Leave a row empty to skip it. Then run evals/rank_eval.py.

The logic lives in src/jobs_search/evaluation.py; this script just wires it up.
"""

from jobs_search.embeddings import embed_texts
from jobs_search.evaluation import pick_jobs_to_label, write_label_sheet
from jobs_search.profile import load_profile, profile_dir
from jobs_search.rag import chroma_path, open_index
from jobs_search.store import connect, database_path


def main() -> None:
    conn = connect(database_path())
    index = open_index(chroma_path())
    profile = load_profile(profile_dir())
    sheet_path = profile_dir() / "labeled_jobs.csv"

    jobs = pick_jobs_to_label(conn, index, profile, embed_texts)
    if not jobs:
        print("No jobs found. Fetch and index some jobs first.")
        return

    write_label_sheet(sheet_path, jobs)
    print(f"Wrote {len(jobs)} jobs to {sheet_path}.")
    print("Open it, write good or bad in the `label` column, save, then run:")
    print("    uv run python evals/rank_eval.py")


if __name__ == "__main__":
    main()

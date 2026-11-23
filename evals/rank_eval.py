"""Score the ranking against your labeled jobs, and compare with the last run.

Run from the repo root:

    uv run python evals/rank_eval.py
    uv run python evals/rank_eval.py "rewrote avoid item"    <- optional note

It reads profile/labeled_jobs.csv, prints precision@10 and any good jobs
ranked below bad ones, and adds a row to data/eval_history.csv.

The logic lives in src/jobs_search/evaluation.py; this script just wires it up.
"""

import sys

from jobs_search.embeddings import embed_texts
from jobs_search.evaluation import append_history, evaluate, last_precision, load_labels
from jobs_search.profile import load_profile, profile_dir
from jobs_search.rag import chroma_path, open_index
from jobs_search.store import connect, database_path

MIN_LABELS = 10


def main() -> None:
    labels_path = profile_dir() / "labeled_jobs.csv"
    history_path = database_path().parent / "eval_history.csv"
    note = " ".join(sys.argv[1:])  # anything typed after the script name

    if not labels_path.exists():
        print(
            f"No {labels_path}. Make one with: uv run python evals/make_label_sheet.py"
        )
        return

    labels = load_labels(labels_path)
    good = sum(1 for item in labels if item.label == "good")
    bad = len(labels) - good
    if len(labels) < MIN_LABELS or good == 0 or bad == 0:
        print(
            f"Need at least {MIN_LABELS} labeled jobs, with some good AND some bad. "
            f"So far: {good} good, {bad} bad. Label more rows in {labels_path}."
        )
        return

    conn = connect(database_path())
    index = open_index(chroma_path())
    profile = load_profile(profile_dir())
    report = evaluate(conn, index, profile, embed_texts, labels)

    previous = last_precision(history_path)
    print(f"Labeled jobs scored: {report.good_count} good, {report.bad_count} bad")
    if previous is None:
        print(f"precision@10: {report.precision_at_10:.2f}  (first run)")
    else:
        change = report.precision_at_10 - previous
        print(
            f"precision@10: {report.precision_at_10:.2f}  (last run {previous:.2f}, change {change:+.2f})"
        )
        if change < 0:
            print("  The number went DOWN. Consider undoing the last change.")

    print("\nRanking of your labeled jobs (best score first):")
    for position, item in enumerate(report.ranked, start=1):
        hidden = "" if item.passes_filters else "   [filters would hide this]"
        print(f"  {position:2}. {item.score:.3f}  {item.label:4}  {item.title}{hidden}")

    if report.good_below_bad:
        print("\nGood jobs ranked below bad ones (number of bad jobs above them):")
        for title, count in report.good_below_bad.items():
            print(f"  - {title}: {count}")
    if report.good_filtered_out:
        print(
            "\nGood jobs your filters would drop:", ", ".join(report.good_filtered_out)
        )
    if report.missing_ids:
        print(
            "\nLabeled ids not found in the store/index:", ", ".join(report.missing_ids)
        )

    append_history(history_path, report, note)
    print(f"\nSaved this run to {history_path}")


if __name__ == "__main__":
    main()

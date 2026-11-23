# jobs_search

A personal job-search agent: it watches company job boards, scores new postings against a profile,
has Claude rerank the best ones, and lets you chat with the results. Built as a hands-on project
for learning agents and RAG.

## Setup

```bash
uv sync
```

Then create a `.env` file in this folder containing `ANTHROPIC_API_KEY=<your key>`.

Run every command from this folder (the repo root).

## Commands

```bash
uv run jobs-search discover   # find companies with job boards -> companies.csv
uv run jobs-search fetch      # download, save and index their jobs
uv run jobs-search match      # your best matches (embeddings only, free)
uv run jobs-search review     # Claude reranks the best matches (costs a little)
uv run jobs-search chat       # talk to the job agent (costs a little)
uv run jobs-search label      # make a sheet of jobs to label good/bad
uv run jobs-search eval       # score the ranking against your labels
```

Add `--help` to any command to see its options. `docs/tutorial.pdf` explains how it all works.

## Layout

| Path | What | In git? |
| --- | --- | --- |
| `CLAUDE.md`, `skills.md` | Instructions and skills for Claude Code, the coding assistant | yes |
| `agent_prompts/` | Instructions and skills for the job agent, sent to Claude at runtime | yes |
| `src/jobs_search/` | The Python package | yes |
| `profile/` | Your profile (`profile.toml`) and `resumes/` | no |
| `tests/` | Tests; `tests/fixtures/profile/` is a made-up profile that also shows the format | yes |
| `companies.csv` | Job boards to fetch, built and pruned by `discover` and `fetch` (not personal) | yes |
| `company_seeds.txt` | Starter guesses for `discover`; each is checked before use | yes |
| `docs/` | The tutorial PDF and the script that builds it | yes |
| `data/` | SQLite database (`jobs.db`), the Chroma index and the embedding model. Set `JOBS_SEARCH_DATA_DIR` to move it | no |
| `.env` | API keys | no |

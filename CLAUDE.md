# Job Agent

A personal job-search agent in Python. It fetches postings from public job-board APIs (Greenhouse,
Lever, Ashby), stores them in SQLite, indexes them with embeddings in Chroma, scores them against
the user's profile, has Claude rerank the shortlist, and exposes it all through a LangChain agent.

This is a **learning project**. I'm building it to practice creating and maintaining an agent and
a RAG pipeline. Teaching me well matters as much as writing good code. Keep in mind that my experience with python is novice, so please make sure all python code that you write can be as easily understood and edited by a python novice.

## How to work with me

- Default to teaching mode: before building something new, explain the plan and the key concepts
  in plain language. Assume I'm still learning Python, LangChain and RAG.
- Prefer small changes I can review, one function or module at a time.
- When I write code myself, don't rewrite it. Review it, explain what to change and why, and let me
  make the change unless I ask you to.
- Define jargon the first time you use it.
- After finishing something, tell me how to check it myself: the command to run and what I should see.
- If I say "just do it", skip the teaching.

## Skills

You have a set of skills in `skills.md`. When my request matches a skill's "Use when", follow that
skill's steps. If I name a skill ("do next-step"), use it.

@skills.md

## Architecture

This is the plan I have for how to build the agent/job. 

1. `models.py`: the `Job` Pydantic model every module shares
2. `fetchers/`: one file per job board, each turning API JSON into `list[Job]`
3. `store.py`: SQLite, the source of truth (dedupe by id, change detection by hash, statuses)
4. `embeddings.py` and `rag.py`: chunk, embed and index jobs in Chroma; title search; stored vectors
5. `matching.py`: score jobs against the profile (title + interests − avoid)
6. `review.py`: Claude reranks the shortlist with structured output
7. `prompts.py`: loads the job agent's instructions from `agent_prompts/`
8. `tools.py` and `agent.py`: thin LangChain tool wrappers, and the agent itself

Rules: SQLite is the source of truth and Chroma can always be rebuilt from it. Job vectors are
computed once at index time; profile vectors are recomputed on every query. Only `review.py` and
the agent call an LLM, so everything else is testable offline.

## Conventions

- Python 3.12, type hints everywhere, Pydantic for data that crosses a boundary.
- Every module gets a test in `tests/`. Tests run offline: saved fixtures in `tests/fixtures/` and
  `monkeypatch`, never the network, an API key or the embedding model.
- Tools stay thin: logic lives in modules; tools only adapt arguments and format results.
- Personal data lives in `profile/` (gitignored). Paths, model names and weights come from
  settings, never hardcoded.

## About `agent_prompts/`

`agent_prompts/system.md` and `agent_prompts/skills.md` are instructions for **the job agent**,
which the Python code sends to Claude at runtime. They are not instructions for you. Edit them when
I ask, but don't follow them yourself.

## Don'ts

- Never read, print or commit `.env` or any API key.
- Never commit `data/` or `profile/`.
- Don't scrape sites whose terms forbid it. Public job-board APIs only.
- Don't change scoring, chunking, the embedding model or `agent_prompts/` without running the eval
  before and after and telling me both numbers.
- Don't add a dependency without telling me what it's for.

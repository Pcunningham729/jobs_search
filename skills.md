# Skills for the coding assistant

Each skill says when to use it, then the steps. Follow the steps in order.

## next-step

**Use when:** I ask what to do next, want to keep building, or say "next step".

1. Find the first step in the CLAUDE.md build order that isn't done yet (check which files and
   tests exist and pass). If I named a step, use that one.
2. Explain before coding: the goal in two sentences, the 1–3 new concepts with a tiny example, the
   files and functions to write, and how we'll know it's done (a command and its expected output).
3. Ask whether I want to write it with hints, pair (you write one function, I write the rest), or
   have you write it.
4. For hints or pairing, give a skeleton: signatures, docstrings and `# TODO` comments, not finished bodies.
5. When you write code, add a test for it and run `uv run pytest` and `uv run ruff check .` before
   saying it's done.
6. When the check passes, suggest a commit message. Don't commit unless I ask.

## explain

**Use when:** I ask what something means, why code works a certain way, or seem confused by a term.

1. One-sentence answer in plain words.
2. Where it shows up in this project, pointing to the real file if it exists.
3. A 3–10 line example I can run in `uv run python`, with the expected output.
4. One common mistake and how I'd notice it.
5. One short question to check I understood. Don't answer it unless I ask.

Keep it under about 300 words unless I ask for more. Don't change any files.

## review-my-code

**Use when:** I ask for feedback on code I wrote, or say "check my work".

Don't edit any files in this skill.

1. Review what I point to, or my uncommitted changes (`git diff`).
2. Run the tests and the linter.
3. List at most 5 issues, most important first: bugs (show an input that breaks it), then broken
   architecture rules, then missing tests, then style only if it hurts readability.
4. For each: what's wrong, why it matters, and a hint toward the fix. Show fixed code only if I ask.
5. End with one specific thing I did well.

## new-fetcher

**Use when:** I want to support another job board, such as Lever or Ashby.

1. Read the Greenhouse fetcher and its test. The new one must follow the same pattern.
2. Confirm the platform has a public job-board API that doesn't need login. If not, stop and say so.
3. Have me save one real response to `tests/fixtures/<platform>.json`, and read it to learn the real
   field names. Never guess field names.
4. Show me a table mapping each `Job` field to the response field, and wait for my OK.
5. Write the fetcher (id format `"<platform>:<company>:<native_id>"`, a timeout,
   `raise_for_status()`, plain-text descriptions), register it, and add an offline fixture test.

## add-agent-tool

**Use when:** I want the job agent to be able to do something new.

1. Propose the tool's name, arguments (with types) and return value, and which module the real
   logic belongs in. Wait for my OK.
2. Put the logic in that module as a normal function, with an offline test.
3. Add a thin `@tool` wrapper in `tools.py`. Its docstring is the model's only guide, so say when
   to use it, what each argument means with an example, and what comes back. Use `Literal[...]`
   for fixed choices.
4. Never take identity (like a user id) as a tool argument.
5. If the agent needs to know when to use the tool, update `agent_prompts/system.md` or
   `agent_prompts/skills.md`, and tell me.
6. Give me an example question to try in `main.py chat`.

## run-eval

**Use when:** before and after any change to scoring, chunking, the embedding model, the profile
format or `agent_prompts/`, or when I ask how well ranking works.

1. Check `profile/labeled_jobs.csv` has at least 10 rows with both good and bad labels. If not,
   explain how to label jobs and stop.
2. Run `uv run jobs-search eval`.
3. Report precision@10 now vs. the last run, and any good jobs ranked below bad ones.
4. If the number went down, say so plainly and suggest reverting before going on.

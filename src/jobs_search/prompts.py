"""Load the job agent's instructions from agent_prompts/.

The agent's instructions live in plain Markdown files, not in Python, so you
can edit how the agent behaves without touching code:

    agent_prompts/system.md   who the agent is and its rules
    agent_prompts/skills.md   step-by-step recipes, one "## name" section each

A SYSTEM PROMPT is the text sent to Claude before the conversation starts.
It sets the agent's role and rules for the whole chat. agent.py will use
agent_system_prompt() as that text.

Per CLAUDE.md: changing these files changes the agent, so run the eval
before and after you edit them.
"""

import os
from pathlib import Path


def prompts_dir() -> Path:
    """The prompts folder. Override with the JOBS_SEARCH_PROMPTS_DIR environment variable."""
    return Path(os.environ.get("JOBS_SEARCH_PROMPTS_DIR", "agent_prompts"))


def load_prompt(name: str, folder: Path | None = None) -> str:
    """Read one prompt file, e.g. load_prompt("system") reads system.md.

    Raises:
        FileNotFoundError: if there's no such file (the message names the path).
    """
    folder = folder or prompts_dir()
    path = folder / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file at {path}")
    # .strip() drops blank lines at the start and end, so joining files
    # together later doesn't leave big gaps.
    return path.read_text(encoding="utf-8").strip()


def split_skills(text: str) -> dict[str, str]:
    """Split skills.md into {skill name: that skill's full section}.

    A skill starts at a line beginning with "## " (a level-2 Markdown heading)
    and runs until the next one. Anything before the first skill, like the
    "# Skills" title, is ignored.

    >>> split_skills("# Skills\\n## a\\nstep 1\\n## b\\nstep 2")
    {'a': '## a\\nstep 1', 'b': '## b\\nstep 2'}
    """
    skills = {}
    current_name = None  # the skill we're inside right now (None = not in one yet)
    current_lines = []

    for line in text.splitlines():
        if line.startswith("## "):
            # A new heading: save the skill we were collecting, then start the new one.
            if current_name is not None:
                skills[current_name] = "\n".join(current_lines).strip()
            current_name = line[3:].strip()  # the text after "## "
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:  # don't forget the last skill
        skills[current_name] = "\n".join(current_lines).strip()
    return skills


def load_skill(name: str, folder: Path | None = None) -> str:
    """One skill's section from skills.md, e.g. load_skill("screen-posting").

    Raises:
        ValueError: for an unknown skill, listing the ones that exist.
    """
    skills = split_skills(load_prompt("skills", folder))
    if name not in skills:
        available = ", ".join(skills)
        raise ValueError(f"No skill called {name!r}. Skills: {available}")
    return skills[name]


def agent_system_prompt(folder: Path | None = None) -> str:
    """The full system prompt for the chat agent: system.md, then skills.md."""
    return load_prompt("system", folder) + "\n\n" + load_prompt("skills", folder)

"""The job agent: Claude, plus the tools in tools.py, in a chat loop.

An AGENT is a model that decides for itself which tools to call. One turn
goes like this:

    you:     "anything good this week?"
    Claude:  (decides to call find_matches with new_since_days=7)
    tool:    returns a list of jobs as text
    Claude:  (reads it, maybe calls get_job_details too)
    Claude:  "Here are the 3 best..."

LangChain's create_agent() runs that loop for us: send the conversation to
Claude, run whatever tools it asks for, send the results back, and repeat
until Claude answers in plain text.

Start a chat (needs ANTHROPIC_API_KEY in .env; each message costs a little):
    uv run jobs-search chat
"""

import os

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage

from jobs_search.profile import Profile
from jobs_search.prompts import agent_system_prompt
from jobs_search.review import format_profile
from jobs_search.tools import ToolContext, make_tools

DEFAULT_MODEL = "claude-opus-5-5"


def agent_model_name() -> str:
    """Which Claude model runs the agent. Override with JOBS_SEARCH_AGENT_MODEL."""
    return os.environ.get("JOBS_SEARCH_AGENT_MODEL", DEFAULT_MODEL)


def chat_model() -> BaseChatModel:
    """The real Claude model, through LangChain's Anthropic integration."""
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv()  # puts ANTHROPIC_API_KEY from .env where the library looks
    return ChatAnthropic(
        model=agent_model_name(),
        max_tokens=16000,
        # effort = how hard Claude thinks before answering (low to max).
        output_config={"effort": "medium"},
    )


def system_prompt(profile: Profile) -> str:
    """agent_prompts/ (rules and skills), plus the person's profile.

    The profile goes in so the agent can say WHICH interest a job matches
    or which avoid item it hits, as the skills ask it to.
    """
    return (
        agent_system_prompt()
        + "\n\nThe person's profile:\n<profile>\n"
        + format_profile(profile)
        + "\n</profile>"
    )


def build_agent(ctx: ToolContext, model: BaseChatModel | None = None):
    """Put together the model, the tools and the system prompt.

    `model` is only passed in by tests (a fake model, so they stay offline).
    """
    return create_agent(
        model or chat_model(),
        tools=make_tools(ctx),
        system_prompt=system_prompt(ctx.profile),
    )


def ask(
    agent, history: list[AnyMessage], question: str
) -> tuple[str, list[AnyMessage]]:
    """Send one question and return (the agent's answer, the updated history).

    The model remembers nothing between calls, so we send the WHOLE
    conversation every time. `history` is that conversation: every question,
    tool call, tool result and answer so far.
    """
    result = agent.invoke({"messages": history + [HumanMessage(question)]})
    messages = result["messages"]
    # The last message is the agent's final answer. `.text` keeps only the
    # text parts (skipping e.g. the "thinking" parts Claude may include).
    return messages[-1].text, messages


def chat_loop(agent) -> None:
    """A simple chat in the terminal. Type 'quit' (or press Ctrl+C) to stop."""
    history = []
    print(f"Job agent ({agent_model_name()}). Type 'quit' to stop.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            break
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer, history = ask(agent, history, question)
        print(f"\nagent> {answer}")

"""Tests for agent.py.

A real agent needs Claude. Here a fake model plays Claude's part with a
script: first it "decides" to call find_matches, then it gives a final
answer. That checks the wiring (model -> tool -> back to model) offline.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from jobs_search.agent import ask, build_agent, system_prompt
from jobs_search.profile import AvoidItem, Interest, Profile
from jobs_search.rag import open_index
from jobs_search.store import connect
from jobs_search.tools import ToolContext

PROFILE = Profile(
    titles=["data engineer"],
    interests=[Interest(name="pipelines", text="Building data pipelines.")],
    avoid=[AvoidItem(text="Lots of travel.")],
)


class ScriptedModel(GenericFakeChatModel):
    """A fake chat model that replies with the next message from a list.

    Real models are told about the tools with bind_tools(). The script
    already knows what to say, so this one just ignores them.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def test_system_prompt_has_rules_skills_and_profile():
    prompt = system_prompt(PROFILE)

    assert "find_matches" in prompt  # from agent_prompts/system.md
    assert "## screen-posting" in prompt  # from agent_prompts/skills.md
    assert "- pipelines: Building data pipelines." in prompt
    assert "Lots of travel." in prompt


def test_agent_calls_a_tool_then_answers(tmp_path):
    connect(tmp_path / "jobs.db").close()  # an empty store
    ctx = ToolContext(
        db_path=tmp_path / "jobs.db",
        index=open_index(tmp_path / "chroma"),
        profile=PROFILE,
        embed=lambda texts: [[1.0] for _ in texts],
    )
    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "find_matches", "args": {"limit": 3}, "id": "call_1"}
                ],
            ),
            AIMessage(content="Nothing fits well yet."),
        ]
    )
    agent = build_agent(ctx, model=ScriptedModel(messages=script))

    answer, history = ask(agent, [], "anything good?")

    assert answer == "Nothing fits well yet."
    # The history holds: question, tool call, tool result, answer.
    tool_results = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].content.startswith("No saved jobs matched")
    assert len(history) == 4

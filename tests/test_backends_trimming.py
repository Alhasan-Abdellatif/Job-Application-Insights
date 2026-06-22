"""Regression tests for parallel-tool-call trimming in each backend.

Background: Claude (and OpenAI) can emit multiple tool_use / tool_call
blocks in a single assistant turn. Anthropic *strictly* rejects the
next request if any of those blocks lacks a corresponding tool_result
in the following user message. Our single-call loop only dispatches
the first, so without trimming we leave the rest dangling and the next
API call 400s with:

    messages.N: `tool_use` ids were found without `tool_result` blocks
    immediately after: toolu_XXX

These tests stub the SDK clients to return parallel-call responses and
verify that each backend persists ONLY the first tool block in its
internal history.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from job_application_insights.agents.backends import (
    AnthropicToolUseSession,
    OpenAIToolUseSession,
    ToolCallTurn,
    ToolSpec,
)

TOOLS = [
    ToolSpec(
        name="count_applications",
        description="Count applications.",
        parameters={
            "type": "object",
            "properties": {"company": {"type": "string"}},
        },
    ),
]


# ────────────────────────── Anthropic ──────────────────────────


class _FakeAnthropicBlock:
    """Minimal stand-in for an anthropic SDK content block."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _anthropic_response_with_two_tool_uses() -> Any:
    """Build a response carrying two parallel tool_use blocks (Claude's behaviour)."""
    block_a = _FakeAnthropicBlock(
        type="tool_use",
        id="toolu_A",
        name="count_applications",
        input={"company": "GSK", "since": "2026-01-01", "until": "2026-12-31"},
    )
    block_b = _FakeAnthropicBlock(
        type="tool_use",
        id="toolu_B",
        name="count_applications",
        input={"company": "GSK", "since": "2025-01-01", "until": "2025-12-31"},
    )
    response = MagicMock()
    response.content = [block_a, block_b]
    return response


def test_anthropic_trims_to_first_tool_use_on_parallel_response() -> None:
    """Two parallel tool_use blocks → history retains only the first.

    This is the exact bug class behind the 400 the user hit on
    "How many GSK applications in 2026 vs 2025?".
    """
    session = AnthropicToolUseSession(system="sys", tools=TOOLS, api_key="dummy")
    session._client = MagicMock()
    session._client.messages.create.return_value = _anthropic_response_with_two_tool_uses()

    turn = session.submit_question("How many GSK in 2026 vs 2025?")

    # Loop dispatches the first call.
    assert isinstance(turn, ToolCallTurn)
    assert turn.args["since"] == "2026-01-01"
    assert session._last_tool_use_id == "toolu_A"

    # History must contain ONLY the first tool_use block. If both were
    # persisted, the next request to Anthropic would 400.
    assistant_msg = session._messages[1]  # [0]=user question, [1]=assistant reply
    assert assistant_msg["role"] == "assistant"
    tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["id"] == "toolu_A"


def test_anthropic_keeps_text_block_before_tool_use() -> None:
    """A text "I'll do X" preamble before the tool_use is preserved."""
    text_block = _FakeAnthropicBlock(type="text", text="Let me count.")
    tool_block = _FakeAnthropicBlock(
        type="tool_use", id="toolu_X", name="count_applications", input={}
    )
    response = MagicMock()
    response.content = [text_block, tool_block]

    session = AnthropicToolUseSession(system="sys", tools=TOOLS, api_key="dummy")
    session._client = MagicMock()
    session._client.messages.create.return_value = response
    session.submit_question("q")

    blocks = session._messages[1]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]


# ────────────────────────── OpenAI ──────────────────────────


def _openai_response_with_two_tool_calls() -> Any:
    """Build a response carrying two parallel tool_calls."""
    call_a = MagicMock()
    call_a.id = "call_A"
    call_a.type = "function"
    call_a.function.name = "count_applications"
    call_a.function.arguments = '{"company": "GSK", "since": "2026-01-01", "until": "2026-12-31"}'
    call_a.model_dump.return_value = {
        "id": "call_A",
        "type": "function",
        "function": {
            "name": "count_applications",
            "arguments": ('{"company": "GSK", "since": "2026-01-01", "until": "2026-12-31"}'),
        },
    }
    call_b = MagicMock()
    call_b.id = "call_B"
    call_b.type = "function"
    call_b.function.name = "count_applications"
    call_b.function.arguments = '{"company": "GSK", "since": "2025-01-01", "until": "2025-12-31"}'
    call_b.model_dump.return_value = {
        "id": "call_B",
        "type": "function",
        "function": {
            "name": "count_applications",
            "arguments": ('{"company": "GSK", "since": "2025-01-01", "until": "2025-12-31"}'),
        },
    }
    msg = MagicMock()
    msg.tool_calls = [call_a, call_b]
    msg.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [call_a.model_dump.return_value, call_b.model_dump.return_value],
    }
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def test_openai_trims_to_first_tool_call_on_parallel_response() -> None:
    """Two parallel tool_calls → history retains only the first."""
    session = OpenAIToolUseSession(system="sys", tools=TOOLS, api_key="dummy")
    session._client = MagicMock()
    session._client.chat.completions.create.return_value = _openai_response_with_two_tool_calls()

    turn = session.submit_question("How many GSK in 2026 vs 2025?")

    assert isinstance(turn, ToolCallTurn)
    assert turn.args["since"] == "2026-01-01"
    assert session._last_tool_call_id == "call_A"

    # The persisted assistant message contains only ONE tool_call.
    # messages[0] is the system prompt; [1] is the user question; [2] is
    # the assistant reply we just persisted.
    assistant_msg = session._messages[2]
    assert assistant_msg.get("role") == "assistant"
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "call_A"

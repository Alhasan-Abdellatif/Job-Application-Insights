"""Provider-agnostic tool-use sessions.

Each LLM provider exposes function calling slightly differently —
Gemini uses ``function_call``/``function_response`` parts in a
``contents`` list, Anthropic uses ``tool_use``/``tool_result`` content
blocks in a ``messages`` list, OpenAI uses ``tool_calls`` arrays with
``tool_call_id`` correlation. The :class:`ToolUseSession` Protocol
hides those differences; the loop in :mod:`tool_use` drives sessions
as a state machine.

Concrete sessions:

* :class:`GeminiToolUseSession` — wraps ``google-genai``.
* :class:`AnthropicToolUseSession` — wraps ``anthropic``.
* :class:`OpenAIToolUseSession` — wraps ``openai``.

Plus a :func:`make_tool_use_session` factory keyed on a provider name,
and :class:`StubToolUseSession` for tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.generate import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
)

AGENT_PROVIDER_NAMES: tuple[str, ...] = ("gemini", "anthropic", "openai")


# ────────────────────────── shared data types ──────────────────────────


class ToolSpec(BaseModel):
    """Provider-agnostic declaration of one callable tool.

    ``parameters`` is a JSON-schema-style dict (``type: "object"`` with
    ``properties`` and ``required`` keys). All three SDKs accept this
    format directly; each backend converts to its native type at
    submission time.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass(frozen=True)
class TextTurn:
    """The model produced a final text answer — the loop stops."""

    text: str


@dataclass(frozen=True)
class ToolCallTurn:
    """The model asked to call a tool; the loop dispatches and continues."""

    name: str
    args: dict[str, Any]


# A turn returned by a session.
ToolUseTurn = TextTurn | ToolCallTurn


class ToolUseSession(Protocol):
    """Stateful conversation that drives one tool-use loop turn at a time.

    A fresh session is created per question. The loop alternates between
    :meth:`submit_question` (first user turn) / :meth:`submit_tool_result`
    (continuation after a tool call) and reading back a :class:`TextTurn`
    (terminal) or :class:`ToolCallTurn` (needs follow-up).
    """

    def submit_question(self, question: str) -> ToolUseTurn: ...

    def submit_tool_result(self, name: str, result: dict[str, Any]) -> ToolUseTurn: ...


# ────────────────────────── Gemini backend ──────────────────────────


class GeminiToolUseSession:
    """Tool-use session backed by Google's ``google-genai`` SDK."""

    def __init__(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._contents: list[genai_types.Content] = []
        declarations = [
            genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=self._to_schema(tool.parameters),
            )
            for tool in tools
        ]
        self._config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            tools=[genai_types.Tool(function_declarations=declarations)],
        )

    @staticmethod
    def _to_schema(parameters: dict[str, Any]) -> genai_types.Schema:
        """Translate the generic JSON-schema dict to a genai Schema."""
        prop_specs = parameters.get("properties", {})
        properties = {
            prop_name: genai_types.Schema(
                type=getattr(genai_types.Type, str(spec.get("type", "STRING")).upper()),
                description=spec.get("description"),
            )
            for prop_name, spec in prop_specs.items()
        }
        return genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties=properties,
            required=list(parameters.get("required", [])),
        )

    def submit_question(self, question: str) -> ToolUseTurn:
        self._contents.append(
            genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
        )
        return self._step()

    def submit_tool_result(self, name: str, result: dict[str, Any]) -> ToolUseTurn:
        self._contents.append(
            genai_types.Content(
                role="tool",
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(name=name, response=result)
                    )
                ],
            )
        )
        return self._step()

    def _step(self) -> ToolUseTurn:
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._contents,
            config=self._config,
        )
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        parts = getattr(content, "parts", None) or []

        # First check for a function_call; Gemini won't mix text+call in
        # the same response for our config.
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                name = str(getattr(fc, "name", ""))
                args = dict(getattr(fc, "args", None) or {})
                # Preserve the model's call in history so the next turn
                # has the full context.
                if content is not None:
                    self._contents.append(content)
                return ToolCallTurn(name=name, args=args)

        text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
        return TextTurn(text=text)


# ────────────────────────── Anthropic backend ──────────────────────────


class AnthropicToolUseSession:
    """Tool-use session backed by the ``anthropic`` SDK."""

    def __init__(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._system = system
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]
        # We need to remember the last tool_use id so we can correlate
        # the tool_result content block.
        self._last_tool_use_id: str | None = None
        self._messages: list[dict[str, Any]] = []

    def submit_question(self, question: str) -> ToolUseTurn:
        self._messages.append({"role": "user", "content": question})
        return self._step()

    def submit_tool_result(self, name: str, result: dict[str, Any]) -> ToolUseTurn:
        del name  # Anthropic correlates by tool_use_id, not by name.
        if self._last_tool_use_id is None:
            raise RuntimeError("submit_tool_result called before any tool_use")
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self._last_tool_use_id,
                        "content": json.dumps(result),
                    }
                ],
            }
        )
        return self._step()

    def _step(self) -> ToolUseTurn:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=self._system,
            tools=self._tools,  # type: ignore[arg-type]
            messages=self._messages,  # type: ignore[arg-type]
        )
        # Persist the assistant's reply in history so a subsequent
        # tool_result correlates correctly.
        assistant_blocks = [block.model_dump() for block in response.content]
        self._messages.append({"role": "assistant", "content": assistant_blocks})

        for block in response.content:
            if block.type == "tool_use":
                self._last_tool_use_id = block.id
                return ToolCallTurn(name=block.name, args=dict(block.input))

        text_parts = [getattr(b, "text", "") for b in response.content if b.type == "text"]
        return TextTurn(text="".join(text_parts).strip())


# ────────────────────────── OpenAI backend ──────────────────────────


class OpenAIToolUseSession:
    """Tool-use session backed by the ``openai`` chat-completions SDK."""

    def __init__(
        self,
        *,
        system: str,
        tools: list[ToolSpec],
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._last_tool_call_id: str | None = None

    def submit_question(self, question: str) -> ToolUseTurn:
        self._messages.append({"role": "user", "content": question})
        return self._step()

    def submit_tool_result(self, name: str, result: dict[str, Any]) -> ToolUseTurn:
        del name  # OpenAI correlates by tool_call_id.
        if self._last_tool_call_id is None:
            raise RuntimeError("submit_tool_result called before any tool_call")
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": self._last_tool_call_id,
                "content": json.dumps(result),
            }
        )
        return self._step()

    def _step(self) -> ToolUseTurn:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            tools=self._tools,  # type: ignore[arg-type]
            messages=self._messages,  # type: ignore[arg-type]
        )
        msg = response.choices[0].message
        # Persist the assistant's reply (with any tool_calls) so the
        # next tool_result can correlate.
        self._messages.append(msg.model_dump(exclude_none=True))

        if msg.tool_calls:
            call = msg.tool_calls[0]
            # OpenAI's tool_calls union includes a "custom" variant
            # without a .function attribute — we never declare custom
            # tools so this branch should never fire, but the union
            # type-narrows here for mypy.
            if getattr(call, "type", "function") != "function":
                raise RuntimeError(f"unexpected tool_call type {call.type!r}")
            self._last_tool_call_id = call.id
            # mypy can't narrow the union from the type-string check
            # above, so silence the union-attr error explicitly.
            args = json.loads(call.function.arguments or "{}")  # type: ignore[union-attr]
            return ToolCallTurn(name=call.function.name, args=args)  # type: ignore[union-attr]

        return TextTurn(text=(msg.content or "").strip())


# ────────────────────────── Stub for tests ──────────────────────────


class StubToolUseSession:
    """In-memory session that yields scripted turns. Used by tests."""

    def __init__(self, script: list[ToolUseTurn]) -> None:
        self._script = list(script)
        self.questions_seen: list[str] = []
        self.tool_results_seen: list[tuple[str, dict[str, Any]]] = []

    def submit_question(self, question: str) -> ToolUseTurn:
        self.questions_seen.append(question)
        return self._pop()

    def submit_tool_result(self, name: str, result: dict[str, Any]) -> ToolUseTurn:
        self.tool_results_seen.append((name, result))
        return self._pop()

    def _pop(self) -> ToolUseTurn:
        if not self._script:
            raise AssertionError("StubToolUseSession exhausted")
        return self._script.pop(0)


# ────────────────────────── factory ──────────────────────────


def make_tool_use_session(
    provider: str,
    *,
    system: str,
    tools: list[ToolSpec],
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> ToolUseSession:
    """Construct a :class:`ToolUseSession` by provider name.

    ``provider`` must be one of :data:`AGENT_PROVIDER_NAMES`. ``model``
    defaults to a sensible per-provider default if omitted.
    """
    kwargs: dict[str, Any] = {
        "system": system,
        "tools": tools,
        "api_key": api_key,
        "temperature": temperature,
    }
    if provider == "gemini":
        return GeminiToolUseSession(model=model or DEFAULT_GEMINI_MODEL, **kwargs)
    if provider == "anthropic":
        return AnthropicToolUseSession(model=model or DEFAULT_ANTHROPIC_MODEL, **kwargs)
    if provider == "openai":
        return OpenAIToolUseSession(model=model or DEFAULT_OPENAI_MODEL, **kwargs)
    raise ValueError(f"unknown agent provider {provider!r}; expected one of {AGENT_PROVIDER_NAMES}")

"""Provider-agnostic tool-use agent over the structured ``applications`` table.

The four typed tools from :mod:`job_application_insights.structured.tools`
are exposed to *any* of Gemini, Anthropic, or OpenAI via the
:class:`ToolUseSession` abstraction in :mod:`agents.backends`. The loop
here is identical for all three providers — only the session encodes
the SDK-specific message format.

The loop:

    1. ``session.submit_question(question)`` → first turn.
    2. If it's a :class:`TextTurn`, that's the final answer; stop.
    3. If it's a :class:`ToolCallTurn`, dispatch to the matching
       Python function, then ``session.submit_tool_result(...)`` →
       next turn.
    4. Repeat. Bail at ``max_steps`` to bound runaway models.

Hand-written tool specs live in :func:`build_tool_specs`. Descriptions
are *prompt engineering* — the LLM reads them to choose when to call.
Auto-derivation from the Python signatures would technically work but
produce useless descriptions.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Protocol

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.agents.backends import (
    AGENT_PROVIDER_NAMES,
    StubToolUseSession,
    TextTurn,
    ToolCallTurn,
    ToolSpec,
    ToolUseSession,
    make_tool_use_session,
)
from job_application_insights.structured.tools import (
    CompanyCount,
    MonthCount,
    RoleRecord,
    applications_by_month,
    count_applications,
    find_company_role,
    top_companies,
)

TOOL_NAMES: tuple[str, ...] = (
    "count_applications",
    "top_companies",
    "applications_by_month",
    "find_company_role",
)

DEFAULT_MAX_STEPS: int = 4
"""Upper bound on tool-call iterations. Three turns covers single-tool
questions; the fourth is slack for one composed tool call. Beyond that
the agent is almost certainly stuck and we'd rather fail loudly."""

SYSTEM_INSTRUCTION: str = (
    "You answer questions about Alhasan's job application emails using the "
    "tools provided. The tools query a structured table of his application "
    "acknowledgments (company, role, date, sender, subject). Pick a tool, "
    "fill in its arguments, and use the result to write a short factual "
    "answer. If no tool can answer the question, say so directly — do not "
    "guess or fabricate."
)


# ────────────────────────── data model ──────────────────────────


class ToolCall(BaseModel):
    """One ``(tool name, arguments, raw output)`` step the agent took."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


class ToolUseResult(BaseModel):
    """The output of one :meth:`ToolUseAgent.answer` round-trip."""

    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stopped_reason: Literal["final_answer", "max_steps", "unknown_tool"]


class ToolUseAgent(Protocol):
    """Structural type for any tool-use agent over the structured table."""

    def answer(self, question: str) -> ToolUseResult: ...


# ────────────────────────── dispatch + serialise ──────────────────────────


def _coerce_date(value: Any) -> date | None:
    """Best-effort: ``None``/empty → None; ISO string → :class:`date`; pass-through."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def dispatch(
    con: duckdb.DuckDBPyConnection,
    name: str,
    arguments: dict[str, Any],
) -> int | list[CompanyCount] | list[MonthCount] | list[RoleRecord]:
    """Run the named tool with the (LLM-supplied) ``arguments`` dict.

    Raises
    ------
    ValueError
        If ``name`` isn't one of the four tool names.
    """
    if name == "count_applications":
        # Treat empty-string optionals as omitted — LLMs sometimes emit
        # ``{"company": ""}`` for "no filter" instead of leaving the key
        # out. Passing ``""`` through to SQL would match nothing.
        raw_company = arguments.get("company")
        company = raw_company if raw_company else None
        return count_applications(
            con,
            company=company,
            since=_coerce_date(arguments.get("since")),
            until=_coerce_date(arguments.get("until")),
        )
    if name == "top_companies":
        return top_companies(con, n=int(arguments.get("n", 10)))
    if name == "applications_by_month":
        return applications_by_month(con)
    if name == "find_company_role":
        company = arguments.get("company")
        if not company:
            raise ValueError("find_company_role requires a non-empty 'company' argument")
        return find_company_role(con, str(company))
    raise ValueError(f"unknown tool {name!r}; expected one of {TOOL_NAMES}")


def serialise(
    result: int | list[CompanyCount] | list[MonthCount] | list[RoleRecord],
) -> dict[str, Any]:
    """Turn a tool result into the JSON-serialisable dict the LLM expects.

    Scalars wrap in ``{"value": …}``; row lists wrap in ``{"rows": [...]}``.
    The shape is intentionally uniform so the LLM sees a consistent
    function-response schema regardless of which tool ran.
    """
    if isinstance(result, int):
        return {"value": result}
    return {"rows": [item.model_dump(mode="json") for item in result]}


# ────────────────────────── tool specs ──────────────────────────


def build_tool_specs() -> list[ToolSpec]:
    """The four :class:`ToolSpec`s every provider's session translates from.

    Descriptions are deliberately verbose and example-laden because they
    are the only signal the model has for *when* to use each tool. Short
    descriptions led to the model picking the wrong tool in early tests.
    """
    return [
        ToolSpec(
            name="count_applications",
            description=(
                "Count the number of application acknowledgment emails, "
                "optionally filtered by company and/or a date window. Use "
                'this for questions like "How many applications did I send '
                'in 2024?" or "How many GSK applications since March?". '
                "Returns a single integer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": (
                            "Optional company name (case-insensitive exact "
                            'match). Example: "GSK", "MediaTek", '
                            '"University of Edinburgh".'
                        ),
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Optional inclusive lower bound on the email "
                            'date, ISO format YYYY-MM-DD. Example: "2024-01-01".'
                        ),
                    },
                    "until": {
                        "type": "string",
                        "description": (
                            "Optional inclusive upper bound on the email "
                            'date, ISO format YYYY-MM-DD. Example: "2024-12-31".'
                        ),
                    },
                },
            },
        ),
        ToolSpec(
            name="top_companies",
            description=(
                "Return the top-N companies by application count, "
                "descending. Use this for questions like "
                '"Which companies did I apply to most?" or "Top 5 companies '
                'by application volume." Returns a list of '
                "{company, n} rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "How many companies to return. Default 10.",
                    },
                },
            },
        ),
        ToolSpec(
            name="applications_by_month",
            description=(
                "Return the monthly application volume time-series across "
                "the entire corpus, ordered chronologically. Use this for "
                'questions like "In which months did I apply the most?" or '
                '"Show my application volume over time." Returns a list of '
                "{year, month, n} rows."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        ToolSpec(
            name="find_company_role",
            description=(
                "Return the role labels (and citation handles) for every "
                "application to a given company. Use this for questions "
                'like "What role did I apply for at MediaTek?" or "What '
                'positions did I apply to at GSK?". Returns a list of '
                "{doc_id, role} rows. An empty role string means the role "
                "label wasn't extracted from that email — surface that "
                "honestly to the user rather than making one up."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": (
                            "Company name (case-insensitive exact match). " 'Example: "GSK".'
                        ),
                    },
                },
                "required": ["company"],
            },
        ),
    ]


# ────────────────────────── the loop ──────────────────────────


def run_tool_use_loop(
    question: str,
    *,
    con: duckdb.DuckDBPyConnection,
    session: ToolUseSession,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ToolUseResult:
    """Drive a :class:`ToolUseSession` as a state machine.

    Provider-agnostic: ``session`` can be any :class:`ToolUseSession`
    (Gemini, Anthropic, OpenAI, or a stub). The loop dispatches each
    :class:`ToolCallTurn` to :func:`dispatch`, serialises the result,
    and feeds it back via ``session.submit_tool_result``.
    """
    tool_calls: list[ToolCall] = []
    turn = session.submit_question(question)

    for _ in range(max_steps):
        if isinstance(turn, TextTurn):
            return ToolUseResult(
                question=question,
                text=turn.text,
                tool_calls=tool_calls,
                stopped_reason="final_answer",
            )

        # ToolCallTurn — dispatch and continue.
        assert isinstance(turn, ToolCallTurn)
        try:
            raw = dispatch(con, turn.name, turn.args)
        except ValueError as exc:
            return ToolUseResult(
                question=question,
                text=str(exc),
                tool_calls=tool_calls,
                stopped_reason="unknown_tool",
            )

        output = serialise(raw)
        tool_calls.append(ToolCall(name=turn.name, arguments=turn.args, output=output))
        turn = session.submit_tool_result(turn.name, output)

    return ToolUseResult(
        question=question,
        text=turn.text if isinstance(turn, TextTurn) else "",
        tool_calls=tool_calls,
        stopped_reason="max_steps",
    )


# ────────────────────────── live agent ──────────────────────────


class LiveToolUseAgent:
    """Concrete :class:`ToolUseAgent` that builds a fresh session per question.

    Picks the backend by provider name (``gemini`` / ``anthropic`` /
    ``openai``). The DuckDB connection is reused across questions; the
    session — and the conversation it tracks — is per-question.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        provider: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if provider not in AGENT_PROVIDER_NAMES:
            raise ValueError(
                f"unknown agent provider {provider!r}; expected one of {AGENT_PROVIDER_NAMES}"
            )
        self._con = con
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._max_steps = max_steps
        # Pre-build the tool specs once; sessions consume them per-question.
        self._tools = build_tool_specs()

    def answer(self, question: str) -> ToolUseResult:
        if not question.strip():
            raise ValueError("question must be non-empty")
        session = make_tool_use_session(
            self._provider,
            system=SYSTEM_INSTRUCTION,
            tools=self._tools,
            model=self._model,
            api_key=self._api_key,
        )
        return run_tool_use_loop(
            question,
            con=self._con,
            session=session,
            max_steps=self._max_steps,
        )


# Backwards-compatible alias — kept so external scripts that import the
# old name don't break. The default ``provider="gemini"`` preserves the
# Day 3 behaviour.
class GeminiToolUseAgent(LiveToolUseAgent):
    """Tool-use agent locked to Gemini (kept for backwards compatibility)."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        super().__init__(
            con,
            provider="gemini",
            model=model,
            api_key=api_key,
            max_steps=max_steps,
        )


__all__ = [
    "DEFAULT_MAX_STEPS",
    "SYSTEM_INSTRUCTION",
    "TOOL_NAMES",
    "GeminiToolUseAgent",
    "LiveToolUseAgent",
    "StubToolUseSession",
    "ToolCall",
    "ToolUseAgent",
    "ToolUseResult",
    "build_tool_specs",
    "dispatch",
    "run_tool_use_loop",
    "serialise",
]

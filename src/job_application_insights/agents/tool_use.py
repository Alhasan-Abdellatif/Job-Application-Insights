"""Gemini-backed tool-use agent over the structured ``applications`` table.

This is the Day-3 piece: the LLM finally gets to call the four typed
Python tools from :mod:`job_application_insights.structured.tools`.
The agent runs a small loop:

    1. Send the user's question + tool declarations to Gemini.
    2. If Gemini returns a function_call, execute it, append the
       result as a function_response, ask again.
    3. If Gemini returns text, we're done — that's the answer.
    4. Bail at ``max_steps`` so a misbehaving model can't loop forever.

The loop itself (:func:`run_tool_use_loop`) is a free function — it
takes a ``generate_fn`` so tests can drive it with canned Gemini
responses without touching the network.

The four ``FunctionDeclaration``s in :func:`build_function_declarations`
are hand-written, not auto-derived from the Python signatures. The
``description`` field on each is *prompt engineering* (the model reads
it as guidance for *when* to use the tool); auto-derivation would
produce technically correct but useless descriptions.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Protocol

import duckdb
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field

from job_application_insights.generate import DEFAULT_GEMINI_MODEL
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
        # Treat empty-string optionals as omitted — Gemini sometimes
        # emits ``{"company": ""}`` for "no filter" instead of just
        # leaving the key out. Passing ``""`` through to SQL would
        # match nothing.
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
    """Turn a tool result into the JSON-serialisable dict Gemini expects.

    Scalars wrap in ``{"value": …}``; row lists wrap in ``{"rows": [...]}``.
    The shape is intentionally uniform so the LLM sees a consistent
    function-response schema regardless of which tool ran.
    """
    if isinstance(result, int):
        return {"value": result}
    return {"rows": [item.model_dump(mode="json") for item in result]}


# ────────────────────────── function declarations ──────────────────────────


def build_function_declarations() -> list[genai_types.FunctionDeclaration]:
    """The four ``FunctionDeclaration``s Gemini sees.

    Descriptions are deliberately verbose and example-laden because they
    are the only signal the model has for *when* to use each tool. Short
    descriptions led to the model picking the wrong tool in early tests.
    """
    return [
        genai_types.FunctionDeclaration(
            name="count_applications",
            description=(
                "Count the number of application acknowledgment emails, "
                "optionally filtered by company and/or a date window. Use "
                'this for questions like "How many applications did I send '
                'in 2024?" or "How many GSK applications since March?". '
                "Returns a single integer."
            ),
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "company": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "Optional company name (case-insensitive exact "
                            'match). Example: "GSK", "MediaTek", '
                            '"University of Edinburgh".'
                        ),
                    ),
                    "since": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "Optional inclusive lower bound on the email "
                            'date, ISO format YYYY-MM-DD. Example: "2024-01-01".'
                        ),
                    ),
                    "until": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "Optional inclusive upper bound on the email "
                            'date, ISO format YYYY-MM-DD. Example: "2024-12-31".'
                        ),
                    ),
                },
            ),
        ),
        genai_types.FunctionDeclaration(
            name="top_companies",
            description=(
                "Return the top-N companies by application count, "
                "descending. Use this for questions like "
                '"Which companies did I apply to most?" or "Top 5 companies '
                'by application volume." Returns a list of '
                "{company, n} rows."
            ),
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "n": genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description="How many companies to return. Default 10.",
                    ),
                },
            ),
        ),
        genai_types.FunctionDeclaration(
            name="applications_by_month",
            description=(
                "Return the monthly application volume time-series across "
                "the entire corpus, ordered chronologically. Use this for "
                'questions like "In which months did I apply the most?" or '
                '"Show my application volume over time." Returns a list of '
                "{year, month, n} rows."
            ),
            parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={}),
        ),
        genai_types.FunctionDeclaration(
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
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "company": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "Company name (case-insensitive exact match). " 'Example: "GSK".'
                        ),
                    ),
                },
                required=["company"],
            ),
        ),
    ]


# ────────────────────────── the loop ──────────────────────────


def _extract_function_call(response: Any) -> tuple[str, dict[str, Any]] | None:
    """Pull the first function_call out of a Gemini response, or None."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None:
            name = getattr(fc, "name", "")
            args = getattr(fc, "args", None) or {}
            # Gemini's args may be a MapComposite-like proxy; coerce to dict.
            return str(name), dict(args)
    return None


def _extract_text(response: Any) -> str:
    """Pull the concatenated text out of a Gemini response, empty if none."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def run_tool_use_loop(
    question: str,
    *,
    con: duckdb.DuckDBPyConnection,
    generate_fn: Any,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ToolUseResult:
    """Execute the tool-use loop with a stubbable ``generate_fn``.

    ``generate_fn`` is any callable matching
    ``(contents, config) -> response`` — production passes the real
    Gemini ``client.models.generate_content``; tests pass a stub.

    The loop is at most ``max_steps`` iterations. On hitting that ceiling
    we stop and return whatever text we have (often empty); the
    ``stopped_reason`` field records the cause.
    """
    declarations = build_function_declarations()
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        tools=[genai_types.Tool(function_declarations=declarations)],
    )

    contents: list[Any] = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=question)],
        )
    ]
    tool_calls: list[ToolCall] = []

    for _ in range(max_steps):
        response = generate_fn(contents, config)

        call = _extract_function_call(response)
        if call is None:
            text = _extract_text(response)
            return ToolUseResult(
                question=question,
                text=text,
                tool_calls=tool_calls,
                stopped_reason="final_answer",
            )

        name, args = call
        try:
            raw = dispatch(con, name, args)
        except ValueError as exc:
            return ToolUseResult(
                question=question,
                text=str(exc),
                tool_calls=tool_calls,
                stopped_reason="unknown_tool",
            )

        output = serialise(raw)
        tool_calls.append(ToolCall(name=name, arguments=args, output=output))

        # Echo the model's function_call back into history (Gemini
        # requires this so it sees the call it made), then append our
        # function_response so it can continue.
        contents.append(
            genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))
                ],
            )
        )
        contents.append(
            genai_types.Content(
                role="tool",
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(name=name, response=output)
                    )
                ],
            )
        )

    return ToolUseResult(
        question=question,
        text="",
        tool_calls=tool_calls,
        stopped_reason="max_steps",
    )


# ────────────────────────── live Gemini agent ──────────────────────────


class GeminiToolUseAgent:
    """Live Gemini agent implementing :class:`ToolUseAgent`.

    Holds a DuckDB connection, a Gemini client, and the tool-use loop
    parameters. One agent instance can answer many questions in a row;
    each call is a fresh conversation (no carryover state).
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._con = con
        self._model = model
        self._client = genai.Client(api_key=api_key)
        self._max_steps = max_steps

    def answer(self, question: str) -> ToolUseResult:
        """Run one tool-use round-trip; return the typed result."""
        if not question.strip():
            raise ValueError("question must be non-empty")

        def _generate(contents: Any, config: Any) -> Any:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

        return run_tool_use_loop(
            question,
            con=self._con,
            generate_fn=_generate,
            max_steps=self._max_steps,
        )

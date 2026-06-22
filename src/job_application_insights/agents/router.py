"""Question router: classify each question by which engine should answer.

One small LLM call, JSON-structured output, three buckets:

* ``rag`` — the answer lives in the text of some email. Use the Week 2
  retrieve+generate path.
* ``structured`` — the answer is a count / aggregation / top-N / date
  filter. Use the Day 3 tool-use agent.
* ``both`` — the question has two parts of different shapes. Decompose
  into a ``structured_question`` and a ``rag_question``; the
  orchestrator will run each engine on its sub-question and compose.

Provider-agnostic: the router takes any :class:`LLMClient` (Gemini,
Anthropic, OpenAI) and asks it to return JSON. Each provider follows
the JSON-only instruction well enough at temperature 0; for Gemini we
could additionally set ``response_mime_type="application/json"`` but
keep the surface uniform here for portability.

Conservative fallback: anything that *isn't* clean parseable JSON
matching :class:`RouterDecision` becomes ``mode="rag"``. RAG is the
safe default because it always produces *some* answer (even "I don't
know"); a wrong tool route can silently return 0.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from job_application_insights.generate import LLMClient

ROUTER_SYSTEM_INSTRUCTION: str = (
    "You are a question router for a job-application RAG system that has "
    "two engines:\n"
    '  - "rag": retrieves and reads email text. Use for questions whose '
    "answer lives in specific email bodies. Examples:\n"
    '      "Did I apply to ALESAYI?", "What role at MediaTek?",\n'
    '      "Who invited me to interview at GSK?"\n'
    '  - "structured": queries a typed table of application metadata '
    "(company, role, date, sender, subject). Use for counts, "
    "aggregations, top-N, date-range filters. Examples:\n"
    '      "How many applications in 2024?",\n'
    '      "Top 5 companies by application count",\n'
    '      "In which months did I apply the most?"\n'
    '  - "both": the question has two parts, one of each kind. Decompose '
    "it into a structured sub-question and a rag sub-question. Example:\n"
    '      "How many GSK applications and what role did I apply for?" =>\n'
    '         structured: "How many GSK applications did I send?"\n'
    '         rag:        "What role did I apply for at GSK?"\n'
    "\n"
    "Return JSON ONLY, matching one of these schemas exactly:\n"
    '  {"mode": "rag"}\n'
    '  {"mode": "structured"}\n'
    '  {"mode": "both", "structured_question": "...", "rag_question": "..."}\n'
    "No prose, no markdown fence, no explanation."
)


class RouterDecision(BaseModel):
    """Typed output of the router.

    ``structured_question`` and ``rag_question`` are populated only when
    ``mode == "both"``. A model validator enforces that — if the LLM
    emits inconsistent JSON we want a :class:`ValidationError` at the
    boundary, not silent corruption downstream.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["rag", "structured", "both"]
    structured_question: str | None = None
    rag_question: str | None = None

    def model_post_init(self, _context: Any) -> None:
        if self.mode == "both":
            if not self.structured_question or not self.rag_question:
                raise ValueError("mode='both' requires both structured_question and rag_question")
        # Disallow the sub-questions on single-engine modes — keeps the
        # downstream orchestrator simple (mode tells you everything).
        elif self.structured_question or self.rag_question:
            raise ValueError(f"mode={self.mode!r} must not carry sub-questions")


def _parse_router_json(raw: str) -> RouterDecision | None:
    """Best-effort: strip ```json fences, parse, validate.

    Returns ``None`` on any failure (bad JSON, missing fields, wrong
    types). The caller falls back to ``mode="rag"`` on ``None``.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip a markdown fence: ```json\n...\n```  or  ```\n...\n```
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    try:
        return RouterDecision.model_validate(obj)
    except ValidationError:
        return None


def classify(question: str, *, llm_client: LLMClient) -> RouterDecision:
    """Classify a question into one of the three router buckets.

    ``llm_client`` is any :class:`LLMClient` implementation — Gemini,
    Anthropic, OpenAI, or a test stub. The classifier issues one
    ``complete()`` call at ``temperature=0`` and parses the output as
    JSON.

    Conservative fallback: any failure to parse a valid
    :class:`RouterDecision` becomes ``RouterDecision(mode="rag")``.
    """
    if not question.strip():
        raise ValueError("question must be non-empty")

    raw = llm_client.complete(
        system=ROUTER_SYSTEM_INSTRUCTION,
        user=question,
        temperature=0.0,
    )
    decision = _parse_router_json(raw)
    if decision is None:
        return RouterDecision(mode="rag")
    return decision

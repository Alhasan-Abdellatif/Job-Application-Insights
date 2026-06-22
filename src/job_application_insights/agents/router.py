"""Question router: classify each question by which engine should answer.

One small Gemini call, JSON-structured output, three buckets:

* ``rag`` — the answer lives in the text of some email. Use the Week 2
  retrieve+generate path.
* ``structured`` — the answer is a count / aggregation / top-N / date
  filter. Use the Day 3 tool-use agent.
* ``both`` — the question has two parts of different shapes. Decompose
  into a ``structured_question`` and a ``rag_question``; the
  orchestrator will run each engine on its sub-question and compose.

Conservative fallback: anything that *isn't* clean parseable JSON
matching :class:`RouterDecision` becomes ``mode="rag"``. RAG is the
safe default because it always produces *some* answer (even if "I
don't know"); a wrong tool route can silently return 0.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, ValidationError

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


def _extract_text(response: Any) -> str:
    """Pull concatenated text out of a Gemini response — empty on misses."""
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


def classify(question: str, *, generate_fn: Any) -> RouterDecision:
    """Classify a question into one of the three router buckets.

    ``generate_fn`` is the same callable shape as the tool-use loop:
    ``(contents, config) -> response``. Tests pass a stub; production
    passes the real ``client.models.generate_content`` wrapped.

    Conservative fallback: any failure to parse a valid RouterDecision
    becomes ``RouterDecision(mode="rag")``.
    """
    if not question.strip():
        raise ValueError("question must be non-empty")

    config = genai_types.GenerateContentConfig(
        system_instruction=ROUTER_SYSTEM_INSTRUCTION,
        temperature=0.0,
        response_mime_type="application/json",
    )
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=question)],
        )
    ]
    response = generate_fn(contents, config)
    raw = _extract_text(response)
    decision = _parse_router_json(raw)
    if decision is None:
        return RouterDecision(mode="rag")
    return decision

"""Augmented generation — turn a question + retrieved chunks into a cited answer.

This module is the **G** in RAG. Everything before it (parse → chunk → embed
→ store → retrieve) was about finding the right context. This module hands
that context to an LLM and gets back natural language.

What it provides:

* :class:`Answer` and :class:`Citation` — Pydantic models for the typed
  output. The whole module is structured so callers always work with these
  rather than raw strings + raw chunks.
* :class:`LLMClient` Protocol — the minimum interface a language-model
  client must satisfy. Two implementations:

  * :class:`AnthropicClient` for real Claude API calls (production).
  * :class:`EchoClient` for tests and offline development — deterministic,
    no network, no API key.
* :func:`format_prompt` — produces the ``(system, user)`` strings that get
  handed to the LLM. Exposed as a pure function so callers can inspect, log,
  or pre-evaluate the prompt.
* :func:`generate_answer` — the one-call convenience: query + retrieved
  results + client → typed Answer.

Prompt design (worth understanding before tweaking):

* The system message sets the role and the rules. The most important rule
  is *"If the context does not contain the answer, say 'I don't know'."* —
  without it, LLMs cheerfully hallucinate.
* The user message wraps both the question and the context in XML tags
  (``<question>``, ``<context>``). Anthropic's docs specifically recommend
  this pattern; Claude parses it more reliably than free-form text.
* Each retrieved chunk is prefixed with ``[source: <chunk_id>]`` so the
  LLM has an obvious citation anchor. We instruct it to use that exact
  bracketed form so we can parse it back later if we want to.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol

from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from job_application_insights.retrieval.vector_store import RetrievalResult


# ────────────────────────────── constants ──────────────────────────────


SYSTEM_PROMPT: str = (
    "You are an assistant that answers questions about Alhasan's job "
    "application emails. Use ONLY the provided context to answer. If the "
    'context does not contain the answer, say "I don\'t know" rather than '
    "guessing. When you reference information from the context, cite the "
    "source using its bracketed ID (e.g. [source: msg_000123])."
)

DEFAULT_ANTHROPIC_MODEL: str = "claude-haiku-4-5"
"""Cheapest Claude model — fine for dev. Swap to ``claude-sonnet-4-6`` for
demo / final eval runs."""

DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
"""Cheap modern OpenAI model. Swap to ``gpt-4o`` for higher quality."""

DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
"""Google's cheap fast tier. Swap to ``gemini-2.5-pro`` for higher quality."""

DEFAULT_MAX_TOKENS: int = 1024
"""Upper bound on the answer length. RAG answers are usually short; 1024 is
generous and protects against runaway generations."""

DEFAULT_TEMPERATURE: float = 0.0
"""Zero temperature for RAG — we want the most likely answer, not a creative
one. Verifiable, reproducible, citeable."""

_SNIPPET_CHARS: int = 200
"""Characters of chunk text to keep in each Citation snippet."""


# ────────────────────────────── data model ──────────────────────────────


class Citation(BaseModel):
    """A pointer from an Answer back to one retrieved chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=-1.0, le=1.0)
    snippet: str = Field(default="")


class Answer(BaseModel):
    """The output of a RAG round-trip — text + the chunks it was built from.

    Notes
    -----
    ``citations`` contains every chunk the retriever surfaced for the query,
    not just the ones the LLM literally referenced. This makes the answer
    auditable end-to-end: a reviewer can compare what was retrieved against
    what the LLM produced.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1)
    text: str
    citations: list[Citation]


# ────────────────────────────── LLM clients ──────────────────────────────


class LLMClient(Protocol):
    """Minimum interface a language-model client must implement.

    Using a :class:`typing.Protocol` rather than an abstract base class lets
    third-party clients satisfy the contract by *structural* typing — no
    inheritance required. Drop in any object with this signature and the rest
    of the pipeline works.
    """

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str: ...


class AnthropicClient:
    """Claude-backed implementation of :class:`LLMClient`.

    The Anthropic SDK reads ``ANTHROPIC_API_KEY`` from the environment
    automatically when ``api_key`` is left ``None``. Add it to ``.env`` once
    and forget about it.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Claude responses come back as a list of content blocks; for the
        # text-only single-turn calls we make here, the first block carries
        # the whole answer.
        first = response.content[0]
        text = getattr(first, "text", "")
        return str(text)


class OpenAIClient:
    """OpenAI-backed implementation of :class:`LLMClient`.

    The OpenAI SDK reads ``OPENAI_API_KEY`` from the environment when
    ``api_key`` is left ``None``.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # OpenAI's chat-completions API can theoretically return content=None
        # when a function call was requested instead of text; we don't use
        # tools here so this is defensive.
        content = response.choices[0].message.content
        return content or ""


class GeminiClient:
    """Google Gemini-backed implementation of :class:`LLMClient`.

    The google-genai SDK reads ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``)
    from the environment when ``api_key`` is left ``None``.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        # google-genai exposes a `.text` convenience property that may be
        # None if the model returned a tool call or was blocked by safety.
        return response.text or ""


class EchoClient:
    """Test / offline implementation — no network, no API key, deterministic.

    Extracts the ``[source: …]`` IDs out of the user message and echoes them
    back. Lets us exercise the full RAG loop in tests without burning credits.
    """

    def __init__(self, prefix: str = "ECHO") -> None:
        self.prefix = prefix

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        # `system`, `max_tokens`, `temperature` are intentionally unused here:
        # the EchoClient is a deterministic stand-in for a real LLM.
        del system, max_tokens, temperature
        ids = re.findall(r"\[source:\s*([^\]]+)\]", user)
        if not ids:
            return f"{self.prefix}: no sources found in prompt"
        return f"{self.prefix}: answer derived from {len(ids)} sources: " + ", ".join(ids)


# ────────────────────────────── factory ──────────────────────────────

PROVIDER_NAMES: tuple[str, ...] = ("anthropic", "openai", "gemini", "echo")
"""Recognised provider names for :func:`make_llm_client`."""


def make_llm_client(provider: str = "anthropic", **kwargs: Any) -> LLMClient:
    """Construct an :class:`LLMClient` by provider name.

    Parameters
    ----------
    provider
        One of ``"anthropic"``, ``"openai"``, ``"gemini"``, ``"echo"``.
    **kwargs
        Forwarded to the chosen client's constructor — e.g. ``model``,
        ``api_key`` for the API-backed clients, ``prefix`` for ``EchoClient``.

    Raises
    ------
    ValueError
        If ``provider`` is not one of the recognised names.
    """
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "openai":
        return OpenAIClient(**kwargs)
    if provider == "gemini":
        return GeminiClient(**kwargs)
    if provider == "echo":
        return EchoClient(**kwargs)
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDER_NAMES}")


# ────────────────────────────── public API ──────────────────────────────


def format_prompt(query: str, results: list[RetrievalResult]) -> tuple[str, str]:
    """Build the ``(system, user)`` prompt pair for a RAG query.

    Pure function — no side effects, no LLM call. Useful in eval pipelines
    where you want to inspect the prompt before (or instead of) generation.
    """
    parts: list[str] = [f"<question>\n{query}\n</question>", ""]

    if results:
        parts.append("<context>")
        for result in results:
            parts.append(f"[source: {result.chunk.chunk_id}]")
            parts.append(result.chunk.text)
            parts.append("")  # blank line between chunks for readability
        parts.append("</context>")
    else:
        parts.append("<context>(no context retrieved)</context>")

    parts.append("")
    parts.append(
        "Answer the question using ONLY the context above. Cite sources by their bracketed ID."
    )
    return SYSTEM_PROMPT, "\n".join(parts)


def generate_answer(
    query: str,
    results: list[RetrievalResult],
    client: LLMClient,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Answer:
    """One-shot RAG round-trip: query + chunks → typed :class:`Answer`.

    Parameters
    ----------
    query
        The user's natural-language question. Must be non-empty.
    results
        Retrieved chunks from :meth:`VectorStore.query`, ordered by descending
        score. Empty list is allowed — the LLM will be told there's no context
        and is expected to reply "I don't know."
    client
        Any object satisfying the :class:`LLMClient` protocol.
    max_tokens, temperature
        Forwarded to the LLM call.

    Returns
    -------
    A :class:`Answer` carrying the LLM text plus a citation list (one
    :class:`Citation` per retrieved chunk).
    """
    if not query.strip():
        raise ValueError("query must be non-empty")

    system, user = format_prompt(query, results)
    text = client.complete(
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    citations = [
        Citation(
            chunk_id=result.chunk.chunk_id,
            doc_id=result.chunk.doc_id,
            score=result.score,
            snippet=result.chunk.text[:_SNIPPET_CHARS],
        )
        for result in results
    ]
    return Answer(query=query, text=text.strip(), citations=citations)

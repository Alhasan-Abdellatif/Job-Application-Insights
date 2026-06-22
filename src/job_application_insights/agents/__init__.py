"""Agentic layer over the structured engine + RAG retriever.

* :mod:`tool_use` (Day 3) — lets the LLM call the four structured
  tools to answer count / aggregation / top-N questions RAG cannot.
* :mod:`router` (Day 4) — classifies each question by which engine
  should answer it.
* :mod:`orchestrator` (Day 4) — dispatches to RAG, structured, or both
  per the router's decision.
"""

from job_application_insights.agents.orchestrator import (
    COMPOSE_SYSTEM_PROMPT,
    DEFAULT_RETRIEVAL_K,
    AgenticAgent,
    AgenticAnswer,
)
from job_application_insights.agents.router import (
    ROUTER_SYSTEM_INSTRUCTION,
    RouterDecision,
    classify,
)
from job_application_insights.agents.tool_use import (
    DEFAULT_MAX_STEPS,
    SYSTEM_INSTRUCTION,
    TOOL_NAMES,
    GeminiToolUseAgent,
    ToolCall,
    ToolUseAgent,
    ToolUseResult,
    build_function_declarations,
    dispatch,
    run_tool_use_loop,
    serialise,
)

__all__ = [
    "COMPOSE_SYSTEM_PROMPT",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_RETRIEVAL_K",
    "ROUTER_SYSTEM_INSTRUCTION",
    "SYSTEM_INSTRUCTION",
    "TOOL_NAMES",
    "AgenticAgent",
    "AgenticAnswer",
    "GeminiToolUseAgent",
    "RouterDecision",
    "ToolCall",
    "ToolUseAgent",
    "ToolUseResult",
    "build_function_declarations",
    "classify",
    "dispatch",
    "run_tool_use_loop",
    "serialise",
]

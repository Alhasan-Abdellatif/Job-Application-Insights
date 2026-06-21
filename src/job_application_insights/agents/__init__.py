"""Agentic layer over the structured engine + RAG retriever.

Day 3 — :mod:`tool_use` — lets the LLM call the four structured tools
to answer count / aggregation / top-N questions RAG cannot. Day 4 will
add the question router that picks between this agent and the Week 2
RAG retriever.
"""

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
    "DEFAULT_MAX_STEPS",
    "SYSTEM_INSTRUCTION",
    "TOOL_NAMES",
    "GeminiToolUseAgent",
    "ToolCall",
    "ToolUseAgent",
    "ToolUseResult",
    "build_function_declarations",
    "dispatch",
    "run_tool_use_loop",
    "serialise",
]

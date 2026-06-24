"""FastAPI HTTP wrapper around the agent.

Exposes the Week 1-3 pipeline (RAG / structured / agentic) as a typed
HTTP service. Run with::

    uv run uvicorn job_application_insights.api.main:app --reload

Then open http://localhost:8000/docs for the auto-generated API
explorer, or POST to ``/ask``.
"""

from job_application_insights.api.main import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    ToolCallOut,
    app,
    create_app,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "CitationOut",
    "HealthResponse",
    "ToolCallOut",
    "app",
    "create_app",
]

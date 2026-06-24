# Single image used for both the FastAPI service and the Streamlit demo.
# The CMD is overridden per service in docker-compose.yml.
#
# Build locally:
#   docker build -t jai:dev .
# Or via compose:
#   docker compose build

# ── stage 1: dependency layer ─────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_LINK_MODE=copy

# Install uv (the same package manager we use locally).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

# System deps the heavy ML libs sometimes need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy lockfiles first so `uv sync` is cached when only source changes.
COPY pyproject.toml uv.lock README.md ./
# Empty src tree placeholder so `uv sync` doesn't fail; real source
# lands in the next layer.
RUN mkdir -p src/job_application_insights \
    && touch src/job_application_insights/__init__.py

RUN uv sync --frozen --no-dev --no-install-project

# ── stage 2: app code ─────────────────────────────────────────────────
FROM base AS final

COPY src ./src
COPY streamlit_app.py ./
COPY data ./data

# Install the project itself (now that source is present).
RUN uv sync --frozen --no-dev

# Expose both ports; docker-compose maps the right one per service.
EXPOSE 8000 8501

# Default CMD is the API; the UI service overrides this in compose.
CMD ["uvicorn", "job_application_insights.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

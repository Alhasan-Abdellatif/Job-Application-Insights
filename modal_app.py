"""Modal deployment for the public demo.

Three Modal primitives map our docker-compose stack onto serverless:

* ``modal.Volume`` — persistent disk for the embedded Qdrant index. Survives
  redeploys; populated by the ``ingest`` one-shot below.
* ``@app.cls`` + ``@modal.asgi_app`` — wraps the existing FastAPI app.
  ``@modal.enter`` runs once per container and triggers the same heavy
  state load (embedder, chunks_by_id, DuckDB) that the local lifespan does.
* ``@modal.web_server`` — launches Streamlit as a long-running subprocess
  and proxies its port to a public URL.

Deploy flow (one-time per machine, run from the repo root)::

    uv sync --extra deploy                  # or: pip install -e '.[deploy]'
    modal token new
    modal secret create jai-api-keys \\
        ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GOOGLE_API_KEY=...
    modal run modal_app.py::ingest          # populate the Qdrant volume
    modal deploy modal_app.py               # ship api + ui, prints URLs

After the first ingest, plain ``modal deploy`` ships code changes without
re-ingesting — the volume persists.

The deployed corpus is the **synthetic** ``data/synthetic/demo.csv`` built
by ``scripts/build_synthetic_demo.py`` — never the real ``ack_only.csv``,
which is gitignored and never leaves the dev machine.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "jai"

# ────────────────────────── image ──────────────────────────

# The deployed image carries: project source, the synthetic demo CSV,
# and the Streamlit entrypoint. ``pip_install_from_pyproject`` reuses our
# pinned production deps — no separate requirements file to drift from.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    # `.workdir` and `.env` are build steps and must come BEFORE the
    # `.add_local_*` calls — Modal rejects build steps after local-file
    # additions (cache-invalidation safeguard).
    .workdir("/app")
    .env(
        {
            # Make `import job_application_insights` work for the COPYed src tree.
            "PYTHONPATH": "/app/src",
            # Pin all three runtime knobs the API reads on startup so the
            # container can't fall back to the chroma defaults.
            "JAI_STORE_BACKEND": "qdrant",
            "JAI_QDRANT_PATH": "/data/qdrant",
            "JAI_STRUCTURED_CSV": "/app/data/synthetic/demo.csv",
            # UI: public demo mode hides knobs that would burn money or
            # crash without keys (see streamlit_app.py).
            "JAI_DEMO_MODE": "1",
        }
    )
    .add_local_dir("src", "/app/src")
    .add_local_file("streamlit_app.py", "/app/streamlit_app.py")
    .add_local_file("data/synthetic/demo.csv", "/app/data/synthetic/demo.csv")
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
)

# ────────────────────────── persistence + secrets ──────────────────────────

# Survives across redeploys. Populated once by the ``ingest`` function below;
# the API container mounts it read-only-ish (Qdrant writes its own metadata
# but the chunk corpus is stable between deploys).
qdrant_vol = modal.Volume.from_name("jai-qdrant", create_if_missing=True)

# LLM API keys live in a Modal Secret object. Missing keys mean the
# corresponding provider just won't work; the service starts regardless.
# Pre-create with::
#   modal secret create jai-api-keys ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GOOGLE_API_KEY=...
api_keys_secret = modal.Secret.from_name("jai-api-keys", required_keys=[])

app = modal.App(APP_NAME)


# ────────────────────────── ingest (one-shot) ──────────────────────────


@app.function(
    image=image,
    volumes={"/data": qdrant_vol},
    timeout=600,
)
def ingest() -> None:
    """Populate the Qdrant volume from the synthetic demo CSV.

    Run once via ``modal run modal_app.py::ingest``. Idempotent — re-running
    upserts the same chunk IDs (deterministic UUID5) so the count stays
    stable. Calls into the same ``ingest_command`` that ``jai ingest`` uses,
    so there's exactly one ingestion code path.
    """
    # Deferred — `job_application_insights` only exists *inside* the Modal
    # image, not on the developer's machine when they run `modal deploy`.
    from job_application_insights.cli import ingest_command

    rc = ingest_command(
        [Path("/app/data/synthetic/demo.csv")],
        store_backend="qdrant",
        qdrant_path="/data/qdrant",
    )
    if rc != 0:
        raise RuntimeError(f"ingest failed with exit code {rc}")

    # Commit so a fresh API container sees the new points.
    qdrant_vol.commit()
    print("Qdrant volume committed.")


# ────────────────────────── API (FastAPI as ASGI) ──────────────────────────


@app.cls(
    image=image,
    volumes={"/data": qdrant_vol},
    secrets=[api_keys_secret],
    timeout=300,
    # Scale to zero — first request after idle pays a cold start (~15-30s
    # for the embedder + DuckDB load), subsequent ones are ~200ms.
    min_containers=0,
    # Keep containers alive 10 min between requests so back-to-back
    # questions don't each pay the cold start.
    scaledown_window=600,
)
class APIService:
    """Wraps the FastAPI app behind an ASGI Modal endpoint."""

    @modal.enter()
    def setup(self) -> None:
        """Run once per container — the same heavy load FastAPI's
        lifespan does locally, just hoisted out so Modal can pool it."""
        # Re-load the volume into the local filesystem view (in case
        # another container committed since this one launched).
        qdrant_vol.reload()
        # Deferred — same reason as `ingest`: the package is image-only.
        from job_application_insights.api.main import create_app

        self.fastapi_app = create_app()

    @modal.asgi_app()
    def web(self) -> object:
        return self.fastapi_app


# ────────────────────────── UI (Streamlit subprocess) ──────────────────────────


@app.function(
    image=image,
    secrets=[api_keys_secret],
    timeout=300,
    min_containers=0,
    scaledown_window=600,
)
@modal.web_server(port=8501, startup_timeout=120)
def ui() -> None:
    """Run Streamlit as a long-lived subprocess; Modal proxies port 8501."""
    # Point the UI at the API's Modal endpoint. Modal forbids accessing
    # bound methods on the class directly — you have to instantiate it
    # (no actual container starts here; this is just the proxy handle).
    api_url = APIService().web.get_web_url()  # type: ignore[attr-defined]
    if not api_url:
        raise RuntimeError("APIService.web has no resolved web URL — is it deployed?")
    os.environ["JAI_API_URL"] = api_url

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/app/streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false",
        ],
    )

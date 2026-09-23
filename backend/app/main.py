"""FastAPI app — Phase 8 adapter layer. Every endpoint is a thin wrapper
around already-built, already-tested pure functions from agents/telemetry/
evaluation (Phases 4-7); no business logic lives here. Run from repo root:

    uvicorn backend.app.main:app --reload
"""
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of where uvicorn is launched
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse

from backend.app.api import (
    automations,
    chat,
    comparison,
    connectors,
    equipment,
    evaluations,
    events,
    graph,
    health,
    ingestion,
    knowledge_risk,
    telemetry,
    work_orders,
)
from backend.app.db.database import init_db

# Initialize relational tables
init_db()

app = FastAPI(title="AuRAG Operator Console API")


def cors_origins() -> list[str]:
    configured = os.environ.get("BACKEND_CORS_ORIGINS", "*")
    if not configured or configured.strip() == "*":
        return ["*"]
    # Honor the configured allowlist verbatim. Appending "*" (as before) made
    # every configured restriction a no-op — a silent open door.
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]

# ponytail: wide-open localhost dev origins, no auth — matches the project's
# standing "no auth/permissions" ground rule and this being a local demo app,
# not a deployed multi-tenant service. Both common Next.js dev ports: Next
# falls back to 3001 if 3000 is already taken.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"status": "ok", "service": "AuRAG Operator Console API"}


import logging

logger = logging.getLogger(__name__)


def _error_cors_headers(request: Request) -> dict:
    # CORSMiddleware does not run on exception-handler responses, so echo the
    # request Origin back (safe without credentials) instead of hardcoding "*".
    origin = request.headers.get("origin")
    return {
        "Access-Control-Allow-Origin": origin or "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }


@app.exception_handler(HTTPException)
def flat_http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {"error": str(exc.detail), "detail": str(exc.detail)}
    # Return the real status code. Masking 5xx as 200 (as before) told every
    # monitor and client that failures were successes.
    return JSONResponse(status_code=exc.status_code, content=content, headers=_error_cors_headers(request))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the full traceback server-side only; never leak internals to clients.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An internal error occurred."},
        headers=_error_cors_headers(request),
    )



app.include_router(chat.router, prefix="/api")
app.include_router(equipment.router, prefix="/api")
app.include_router(telemetry.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(ingestion.router, prefix="/api")
app.include_router(knowledge_risk.router, prefix="/api")
app.include_router(comparison.router, prefix="/api")
app.include_router(evaluations.router, prefix="/api")
app.include_router(work_orders.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(connectors.router, prefix="/api")
app.include_router(automations.router, prefix="/api")


# On Render free tier (512MB RAM), pre-warming torch and sentence_transformers
# causes the kernel OOM killer to immediately terminate the instance.
# Models will only be lazy-loaded on demand if requested.


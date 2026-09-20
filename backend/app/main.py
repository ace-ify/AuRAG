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

from fastapi import FastAPI, HTTPException
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
    configured = os.environ.get(
        "BACKEND_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001",
    )
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]

# ponytail: wide-open localhost dev origins, no auth — matches the project's
# standing "no auth/permissions" ground rule and this being a local demo app,
# not a deployed multi-tenant service. Both common Next.js dev ports: Next
# falls back to 3001 if 3000 is already taken.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def flat_http_exception_handler(request, exc: HTTPException):
    # Every endpoint raises HTTPException(..., detail={"error": ..., "detail":
    # ...}) — FastAPI's default handler would wrap that as {"detail": {...}},
    # nesting it one level deeper than documented. This flattens the response
    # to exactly the {"error": ..., "detail": ...} shape README.md describes.
    # If exc.detail is a string or non-dict, defensively wrap it so clients always
    # receive a consistent JSON object structure.
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {"error": str(exc.detail), "detail": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


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

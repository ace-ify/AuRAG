"""FastAPI app — Phase 8 adapter layer. Every endpoint is a thin wrapper
around already-built, already-tested pure functions from agents/telemetry/
evaluation (Phases 4-7); no business logic lives here. Run from repo root:

    uvicorn backend.app.main:app --reload
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import (
    chat,
    comparison,
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
    # to exactly the {"error": ..., "detail": ...} shape README.md describes,
    # for every endpoint at once (one handler, not a per-endpoint fix).
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


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

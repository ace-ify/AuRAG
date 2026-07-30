# AuRAG PRD Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every capability promised by `PRD.md`, wire it through the FastAPI and Next.js product, make runtime state durable, and verify the full system with automated tests and live smoke checks.

**Architecture:** Neo4j remains the domain source of truth. FastAPI exposes focused route modules backed by testable service functions; Redis carries live event fan-out and worker jobs; Qdrant remains the dense retrieval store; mem0 provides cross-session memory when configured. The Next.js workspace uses separate routes for each operator job while a shared layout owns navigation, notifications, and short-lived workspace state.

**Tech Stack:** Python 3.12, FastAPI, Neo4j, Qdrant, Redis/RQ, LangGraph, mem0, RAGAS, pytest; Next.js 16, React 19, Tailwind 4, shadcn/ui, Vitest, Testing Library, Playwright.

---

### Task 1: Reproducible runtime and test foundation

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/conftest.py`
- Create: `backend/app/api/health.py`
- Test: `tests/backend/test_health.py`
- Modify: `backend/app/main.py`
- Modify: `infra/docker-compose.yml`
- Modify: `.env.example`
- Modify: `.gitignore`

- [ ] Add a failing FastAPI test proving `/api/health/live` is process-only and `/api/health/ready` reports Neo4j, Qdrant, Redis, and configured LLM providers independently.
- [ ] Run `pytest tests/backend/test_health.py -q` and confirm the routes are missing.
- [ ] Add unified pinned runtime/dev requirements, health probes, persistent Redis/Qdrant volumes, and documented environment variables.
- [ ] Run the health test and the existing executable validations that do not require external services.
- [ ] Create a trusted Python 3.12 `.venv` with `uv`, install dependencies, and record exact setup commands in `README.md`.

### Task 2: Complete P&ID and OCR ingestion

**Files:**
- Modify: `ingestion/parsers/vision_pnid.py`
- Modify: `ingestion/pipeline.py`
- Modify: `ingestion/parsers/ocr.py`
- Modify: `infra/neo4j/schema.cypher`
- Create: `tests/ingestion/test_pid_connections.py`
- Create: `tests/ingestion/test_ocr_fallback.py`

- [ ] Add failing tests proving P&ID extraction returns connections and ingestion persists `CONNECTED_TO` with `line_type`, source document, and page provenance.
- [ ] Add a failing OCR test proving missing Tesseract cleanly invokes Gemini fallback and reports which engine produced the text.
- [ ] Preserve connection data through `RawChunk`, resolve both endpoints against known equipment, and idempotently `MERGE` relationship provenance.
- [ ] Add explicit unmatched-connection reporting instead of silently dropping diagram edges.
- [ ] Run ingestion unit tests and the existing 13/13 ground-truth validation.

### Task 3: Knowledge-retirement risk capability

**Files:**
- Create: `backend/app/services/knowledge_risk.py`
- Create: `backend/app/api/knowledge_risk.py`
- Modify: `backend/app/main.py`
- Create: `tests/backend/test_knowledge_risk.py`
- Create: `frontend/app/(workspace)/knowledge-risk/page.tsx`
- Create: `frontend/components/knowledge-risk/KnowledgeRiskView.tsx`
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/components/AppSidebar.tsx`

- [ ] Add failing service/API tests for risk ranking, severity thresholds, connected equipment/work orders/failures/documents, coverage gaps, and empty data.
- [ ] Implement `GET /api/knowledge-risk` and `GET /api/knowledge-risk/people/{person_id}` using parameterized Cypher and deterministic risk scoring.
- [ ] Add typed frontend API functions and a route showing summary metrics, ranked personnel, evidence, and capture recommendations.
- [ ] Add loading, empty, error, and mobile states.
- [ ] Run backend tests, frontend tests, lint, and production build.

### Task 4: Honest GraphRAG versus plain-vector-RAG comparison

**Files:**
- Modify: `retrieval/hybrid.py`
- Create: `retrieval/plain_vector.py`
- Create: `backend/app/services/comparison.py`
- Create: `backend/app/api/comparison.py`
- Modify: `backend/app/main.py`
- Create: `tests/retrieval/test_plain_vector.py`
- Create: `tests/backend/test_comparison.py`
- Create: `frontend/app/(workspace)/comparison/page.tsx`
- Create: `frontend/components/comparison/ComparisonWorkspace.tsx`
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/components/AppSidebar.tsx`

- [ ] Add failing tests proving plain RAG uses only dense vector candidates and GraphRAG uses the existing four-source hybrid path.
- [ ] Implement parallel answer generation with shared query, independent contexts/citations, latency, source overlap, and deterministic comparison metadata.
- [ ] Expose `POST /api/comparison`.
- [ ] Build a synchronized two-pane comparison route with raw-context disclosure and truthful error states.
- [ ] Run retrieval/API/frontend tests and build.

### Task 5: Durable RAGAS history and evaluation dashboard

**Files:**
- Replace: `backend/app/core/ragas_jobs.py`
- Create: `backend/app/services/evaluations.py`
- Create: `backend/app/api/evaluations.py`
- Modify: `backend/app/api/chat.py`
- Modify: `backend/app/main.py`
- Modify: `infra/neo4j/schema.cypher`
- Create: `tests/backend/test_evaluations.py`
- Create: `frontend/app/(workspace)/evaluation/page.tsx`
- Create: `frontend/components/evaluation/EvaluationDashboard.tsx`
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/package.json`

- [ ] Add failing tests proving score jobs survive service object recreation, store query/answer/agent/citations/timing, and aggregate by metric/status/agent.
- [ ] Persist `EvaluationRun` nodes in Neo4j while keeping answer delivery fail-open.
- [ ] Expose result, list, and summary endpoints with filters and pagination.
- [ ] Add a dashboard with summary cards, trend visualization, recent results, and low-faithfulness queue.
- [ ] Run score-store/API/frontend tests, lint, and build.

### Task 6: Cross-session memory with mem0

**Files:**
- Create: `backend/app/core/memory.py`
- Modify: `backend/app/api/chat.py`
- Modify: `agents/state.py`
- Modify: `agents/supervisor.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `tests/backend/test_memory.py`

- [ ] Add failing tests for session/user scoped recall, bounded memory injection, explicit expiry metadata, provider failure isolation, and post-answer memory writes.
- [ ] Implement a mem0 adapter configured by `MEM0_API_KEY`, with a no-op unavailable state that is visible in readiness output.
- [ ] Add `session_id` and `user_id` to chat requests, inject recalled context as non-authoritative memory, and write compact answer memories after successful responses.
- [ ] Ensure retrieved plant evidence always outranks memory and citations never point to memory records.
- [ ] Run memory and chat API tests.

### Task 7: Durable proactive events and work-order decisions

**Files:**
- Create: `backend/app/services/events.py`
- Create: `backend/app/services/work_orders.py`
- Create: `backend/app/api/events.py`
- Create: `backend/app/api/work_orders.py`
- Modify: `backend/app/api/telemetry.py`
- Modify: `backend/app/main.py`
- Modify: `infra/neo4j/schema.cypher`
- Create: `telemetry/worker.py`
- Create: `tests/backend/test_events.py`
- Create: `tests/backend/test_work_orders.py`
- Create: `tests/telemetry/test_worker.py`

- [ ] Add failing tests for persisted predictive events, event deduplication, SSE replay via `Last-Event-ID`, and reconnect behavior.
- [ ] Add failing tests for draft creation, edit, accept, reject-with-reason, idempotency, version conflicts, and audit history.
- [ ] Persist `PredictiveEvent`, `Notification`, `WorkOrder`, and `WorkOrderDecision` data in Neo4j and publish notification IDs through Redis.
- [ ] Expose SSE plus polling fallback endpoints.
- [ ] Add a continuous synthetic telemetry worker with configurable interval/equipment set and deterministic demo mode.
- [ ] Run event, work-order, and worker tests.

### Task 8: Route-based professional frontend and workflow wiring

**Files:**
- Create: `frontend/app/(workspace)/layout.tsx`
- Move: existing workspace pages under `frontend/app/(workspace)/`
- Create: `frontend/components/providers/WorkspaceProvider.tsx`
- Create: `frontend/components/notifications/NotificationCenter.tsx`
- Create: `frontend/hooks/use-live-events.ts`
- Create: `frontend/app/(workspace)/work-orders/page.tsx`
- Create: `frontend/app/(workspace)/work-orders/[id]/page.tsx`
- Create: `frontend/components/work-orders/WorkOrderEditor.tsx`
- Modify: `frontend/components/AppSidebar.tsx`
- Modify: `frontend/components/DashboardHeader.tsx`
- Modify: `frontend/components/TelemetryPanel.tsx`
- Modify: `frontend/lib/api.ts`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/test/setup.ts`
- Create: `frontend/e2e/operator-workflows.spec.ts`

- [ ] Add failing component tests for nested-route active state, persistent workspace state, notification deduplication, and work-order decisions.
- [ ] Add a shared workspace layout so sidebar/header/live subscriptions do not remount between routes.
- [ ] Replace local “reviewed/dismissed” simulation with persisted accept/edit/reject actions and audit feedback.
- [ ] Add global live-event notifications and backend connection status.
- [ ] Add route-level loading/error/empty states and responsive layouts.
- [ ] Run Vitest, Playwright smoke tests, lint, and production build.

### Task 9: Deployment and full acceptance verification

**Files:**
- Create: `Dockerfile`
- Create: `render.yaml`
- Create: `frontend/vercel.json`
- Create: `.github/workflows/ci.yml`
- Create: `scripts/bootstrap.ps1`
- Create: `scripts/smoke.ps1`
- Modify: `README.md`
- Modify: `PRD_CLOSURE.md`
- Create: `docs/DEMO_SCRIPT.md`
- Create: `docs/DEPLOYMENT.md`
- Create: `docs/ACCEPTANCE_REPORT.md`

- [ ] Add CI for Python unit tests, frontend tests, lint, and production build.
- [ ] Add backend/worker container definitions and Vercel frontend configuration.
- [ ] Add repeatable bootstrap and smoke scripts covering health, seed, indexing, chat, graph, scoring, retirement risk, comparison, events, and work-order decisions.
- [ ] Verify against local or managed Neo4j, Qdrant, Redis, Groq, Gemini, and mem0 credentials.
- [ ] Record exact evidence for every PRD feature in `docs/ACCEPTANCE_REPORT.md`.
- [ ] Update `PRD_CLOSURE.md` only when each item is proven by current automated or live evidence.

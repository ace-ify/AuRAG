"""POST /api/chat — thin wrapper around agents.supervisor.answer(), the
single entry point spanning all four intents (Phase 5). Its return dict is
already JSON-safe (str/float/bool/list/dict) and already carries citations,
graph_paths, and ragas_scores/ragas_status/low_faithfulness (Phase 6) — no
transform layer needed.

The one thing genuinely new here: agents.supervisor.answer() can raise
uncaught (discovered in Phase 6 verification — a Groq quota exhaustion
inside an agent's own reasoning call propagates all the way up). Every CLI
self-check so far just crashed and that was fine for a terminal; this is the
first place it's exposed to a live UI, so it needs to fail as a clean 503,
not a stack trace on someone's screen."""
from threading import Thread
from uuid import uuid4
import os
import logging

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.core.ragas_jobs import create_score_job, get_score_job, run_score_job
from backend.app.core.memory import get_memory_service
from backend.app.core.neo4j import get_session
from backend.app.core.auth import UserProfile, get_current_user
from backend.app.core.tenant import SiteContext, get_site_context
from agents.guardrails import check_safety_guardrails, mask_pii
from agents.citation_resolver import resolve_sentence_citations

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    user_id: str = "local-operator"
    session_id: str = ""
    site_id: str = ""


def answer_query(
    session,
    query: str,
    memory_context: list[str] | None = None,
    session_id: str | None = None,
    site_id: str | None = None,
) -> dict:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from agents.supervisor import answer

    def _run():
        return answer(
            session,
            query,
            score=False,
            memory_context=memory_context or [],
            session_id=session_id,
            site_id=site_id,
        )

    # Render's reverse proxy hard-kills at 30s; we must respond before that.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=25.0)
        except TimeoutError:
            raise TimeoutError("Agent pipeline exceeded 25s budget")


@router.get("/chat/ping")
@router.post("/chat/ping")
def chat_ping() -> dict:
    return {"status": "pong"}


@router.post("/chat")
def chat(
    request: ChatRequest,
    session=Depends(get_session),
    current_user: UserProfile = Depends(get_current_user),
    site: SiteContext = Depends(get_site_context),
) -> dict:
    session_id = request.session_id or str(uuid4())
    effective_user_id = getattr(current_user, "user_id", request.user_id) if current_user else request.user_id
    effective_site_id = getattr(site, "site_id", request.site_id or "plant-mumbai-01") if site else (request.site_id or "plant-mumbai-01")

    # Safety guardrails — fail CLOSED. If the check itself errors, refuse
    # rather than let an unscreened query through to the agents.
    try:
        guard_check = check_safety_guardrails(request.query)
        guard_ok = guard_check.is_safe
        refusal = guard_check.refusal_message
    except Exception as exc:
        logger.error("Guardrail check errored; failing closed (refusing). %s", exc)
        guard_ok = False
        refusal = (
            "The safety pre-check could not be completed, so this query is refused "
            "as a precaution. Please retry shortly."
        )

    if not guard_ok:
        return {
            "user_query": request.query,
            "routed_agent": "safety_guardrail",
            "agent_response": refusal,
            "citations": [],
            "graph_paths": [],
            "session_id": session_id,
            "site_id": effective_site_id,
            "user_id": effective_user_id,
            "guardrail_status": "REFUSED_SAFETY_VIOLATION",
            "grounding_status": "UNGROUNDED",
            "overall_confidence": 0.0,
            "score_id": None,
            "ragas_status": "skipped_guardrail_refusal",
            "ragas_scores": {},
            "low_faithfulness": False,
        }

    sanitized_query = mask_pii(request.query)

    # Memories (fail-open)
    memories = []
    try:
        memory_service = get_memory_service()
        memories = memory_service.recall(effective_user_id, sanitized_query)
    except Exception as exc:
        logger.warning("Memory recall error: %s", exc)

    # Core reasoning. On failure we ABSTAIN — we must never fabricate a
    # confident answer (the old path returned a canned P-101 compliance answer
    # with invented citations on ANY exception, which for a safety/compliance
    # product is the most dangerous possible failure mode).
    try:
        result = answer_query(
            session,
            sanitized_query,
            memory_context=memories,
            session_id=session_id,
            site_id=effective_site_id,
        )
    except Exception as exc:
        logger.error("Core answer_query failed (%s); abstaining, no fabricated content.", exc, exc_info=True)
        result = {
            "user_query": request.query,
            "routed_agent": "system_degraded",
            "agent_response": (
                "The knowledge system is temporarily unavailable, so I can't produce a "
                "grounded answer right now. Do not treat this as an all-clear — please "
                "retry shortly or consult the source system directly."
            ),
            "citations": [],
            "retrieved_context": [],
            "graph_paths": [],
            "degraded": True,
        }

    # If the graph is degraded and produced no grounded context, abstain rather
    # than let an agent answer from nothing (defensive: the fabrication paths
    # are gone, but empty-context answers on an outage are still misleading).
    # `is True` is deliberate — only a real ResilientNeo4jSession reporting a
    # confirmed outage triggers this, not an arbitrary/mocked session object.
    if getattr(session, "degraded", False) is True and not (result.get("retrieved_context") or []):
        result["degraded"] = True
        if not result.get("citations"):
            result["routed_agent"] = "system_degraded"
            result["agent_response"] = (
                "The knowledge graph is currently unavailable, so I have no grounded "
                "evidence to answer from. Do not treat this as an all-clear — please "
                "retry shortly or consult the source system directly."
            )

    result["session_id"] = session_id
    result["site_id"] = effective_site_id
    result["user_id"] = effective_user_id
    result["memory_recalled"] = len(memories)

    context = result.get("retrieved_context") or []

    # Resolve sentence-level grounding. On error, fail CLOSED: UNGROUNDED with
    # zero confidence — never inflate to GROUNDED/0.95 (the old behavior, which
    # stamped a fabricated answer as verified).
    try:
        grounded = resolve_sentence_citations(
            answer=result["agent_response"],
            context_items=context,
            site_id=effective_site_id,
        )
        result["grounded_claims"] = [c.to_dict() for c in grounded.claims]
        result["grounding_status"] = grounded.grounding_status
        result["overall_confidence"] = grounded.overall_confidence
    except Exception as exc:
        logger.warning("Sentence grounding error; marking UNGROUNDED. %s", exc)
        result["grounded_claims"] = []
        result["grounding_status"] = "UNGROUNDED"
        result["overall_confidence"] = 0.0

    # Non-blocking memory update
    try:
        memory_service = get_memory_service()
        memory_service.remember(
            user_id=effective_user_id,
            session_id=session_id,
            query=sanitized_query,
            answer=result["agent_response"],
        )
    except Exception:
        pass

    # Asynchronous RAGAS scoring (non-blocking, fail-open). Only when there is
    # real retrieved context to score against; scoring a degraded/abstained
    # answer with no context is meaningless. Can be disabled on a
    # memory-constrained host via RAGAS_ASYNC_SCORING=0.
    if context and os.environ.get("RAGAS_ASYNC_SCORING", "1") != "0":
        score_id = create_score_job(
            session,
            query=sanitized_query,
            agent_response=result["agent_response"],
            routed_agent=result.get("routed_agent", ""),
            citations=result.get("citations", []),
            graph_paths=result.get("graph_paths", []),
            retrieved_context=context,
        )
        Thread(
            target=run_score_job,
            args=(score_id, sanitized_query, result["agent_response"], context),
            daemon=True,
        ).start()
        result["score_id"] = score_id
        result["ragas_status"] = "scoring"
    else:
        result["score_id"] = None
        result["ragas_status"] = "skipped_no_context"
    result["ragas_scores"] = {}
    result.setdefault("low_faithfulness", False)
    return result


@router.get("/chat/scores/{score_id}")
def chat_score(score_id: str, session=Depends(get_session)) -> dict:
    job = get_score_job(score_id, session=session)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "score_not_found", "detail": "Unknown score job."})
    return job

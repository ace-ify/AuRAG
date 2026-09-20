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
) -> dict:
    from agents.supervisor import answer

    return answer(
        session,
        query,
        score=False,
        memory_context=memory_context or [],
        session_id=session_id,
    )


@router.post("/chat")
def chat(
    request: ChatRequest,
    session=Depends(get_session),
    current_user: UserProfile = Depends(get_current_user),
    site: SiteContext = Depends(get_site_context),
) -> dict:
    try:
        session_id = request.session_id or str(uuid4())
        if isinstance(current_user, UserProfile):
            effective_user_id = current_user.user_id
        else:
            effective_user_id = request.user_id

        if isinstance(site, SiteContext):
            effective_site_id = site.site_id
        else:
            effective_site_id = request.site_id or "plant-mumbai-01"

        # Safety Guardrails check
        guard_check = check_safety_guardrails(request.query)
        if not guard_check.is_safe:
            return {
                "user_query": request.query,
                "routed_agent": "safety_guardrail",
                "agent_response": guard_check.refusal_message,
                "citations": [],
                "graph_paths": [],
                "session_id": session_id,
                "site_id": effective_site_id,
                "user_id": effective_user_id,
                "guardrail_status": "REFUSED_SAFETY_VIOLATION",
                "score_id": None,
                "ragas_status": "skipped_guardrail_refusal",
                "ragas_scores": {},
                "low_faithfulness": False,
            }

        sanitized_query = mask_pii(request.query)

        memory_service = get_memory_service()
        memories = memory_service.recall(effective_user_id, sanitized_query)
        result = answer_query(
            session,
            sanitized_query,
            memory_context=memories,
            session_id=session_id,
        )
        result["session_id"] = session_id
        result["site_id"] = effective_site_id
        result["user_id"] = effective_user_id
        result["memory_recalled"] = len(memories)

        context = result.get("retrieved_context") or []

        # Resolve sentence-level grounding and authorized deep links
        grounded = resolve_sentence_citations(
            answer=result["agent_response"],
            context_items=context,
            site_id=effective_site_id,
        )
        result["grounded_claims"] = [c.to_dict() for c in grounded.claims]
        result["grounding_status"] = grounded.grounding_status
        result["overall_confidence"] = grounded.overall_confidence

        memory_service.remember(
            user_id=effective_user_id,
            session_id=session_id,
            query=sanitized_query,
            answer=result["agent_response"],
        )
        if context:
            score_id = create_score_job(
                session,
                query=request.query,
                agent_response=result["agent_response"],
                routed_agent=result.get("routed_agent") or result.get("intent") or "unknown",
                citations=result.get("citations") or [],
                graph_paths=result.get("graph_paths") or [],
                retrieved_context=context,
            )
            Thread(
                target=run_score_job,
                args=(
                    score_id,
                    request.query,
                    result["agent_response"],
                    context,
                ),
                daemon=True,
            ).start()
            result["score_id"] = score_id
            result["ragas_status"] = "scoring"
            result["ragas_scores"] = {}
            result["low_faithfulness"] = False
        else:
            result["score_id"] = None
            result["ragas_status"] = "skipped_no_context"
            result["ragas_scores"] = {}
            result["low_faithfulness"] = False
        return result
    except Exception as exc:
        return {
            "user_query": request.query,
            "intent": "general",
            "routing_confidence": 0.0,
            "routed_agent": "system",
            "retrieved_context": [],
            "graph_paths": [],
            "agent_response": f"Service notification: {str(exc)}",
            "citations": [],
            "session_id": session_id if "session_id" in locals() else "",
            "memory_context": [],
            "ragas_scores": {},
            "ragas_status": "error",
            "low_faithfulness": False,
            "site_id": effective_site_id if "effective_site_id" in locals() else "",
            "user_id": effective_user_id if "effective_user_id" in locals() else "",
            "memory_recalled": 0,
            "grounded_claims": [],
            "grounding_status": "ERROR",
            "overall_confidence": 0.0,
            "score_id": None,
            "error": str(exc),
        }



@router.get("/chat/scores/{score_id}")
def chat_score(score_id: str, session=Depends(get_session)) -> dict:
    job = get_score_job(score_id, session=session)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "score_not_found", "detail": "Unknown score job."})
    return job

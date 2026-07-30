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

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    user_id: str = "local-operator"
    session_id: str = ""


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
def chat(request: ChatRequest, session=Depends(get_session)) -> dict:
    try:
        session_id = request.session_id or str(uuid4())
        memory_service = get_memory_service()
        memories = memory_service.recall(request.user_id, request.query)
        result = answer_query(
            session,
            request.query,
            memory_context=memories,
            session_id=session_id,
        )
        result["session_id"] = session_id
        result["memory_recalled"] = len(memories)
        # Memory is part of successful answer delivery, not evaluation.
        # Persist it immediately and fail open inside the adapter so a slow or
        # failed RAGAS job cannot lose cross-session continuity.
        memory_service.remember(
            user_id=request.user_id,
            session_id=session_id,
            query=request.query,
            answer=result["agent_response"],
        )
        context = result.get("retrieved_context") or []
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
        raise HTTPException(status_code=503, detail={"error": "chat_failed", "detail": str(exc)}) from exc


@router.get("/chat/scores/{score_id}")
def chat_score(score_id: str, session=Depends(get_session)) -> dict:
    job = get_score_job(score_id, session=session)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "score_not_found", "detail": "Unknown score job."})
    return job

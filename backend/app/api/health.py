"""Process liveness and dependency readiness endpoints."""

import os

from fastapi import APIRouter, HTTPException

from backend.app.services.health import build_readiness

router = APIRouter()


def _require_env(name: str) -> None:
    if not os.environ.get(name):
        raise RuntimeError(f"{name} is not configured")


def _check_neo4j() -> None:
    from retrieval.index_chunks import get_driver

    get_driver().verify_connectivity()


def _check_qdrant() -> None:
    from retrieval.qdrant_store import get_client

    get_client().get_collections()


def _check_redis() -> None:
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is not configured")
    from redis import Redis

    client = Redis.from_url(
        redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    if client.ping() is not True:
        raise RuntimeError("Redis PING did not return PONG")


def _check_mem0() -> None:
    from backend.app.core.memory import get_memory_service

    status = get_memory_service().status()
    if status["status"] != "up":
        raise RuntimeError(status.get("detail") or status["status"])


def dependency_checks() -> dict:
    return {
        "neo4j": _check_neo4j,
        "qdrant": _check_qdrant,
        "redis": _check_redis,
        "groq": lambda: _require_env("GROQ_API_KEY"),
        "gemini": lambda: _require_env("GEMINI_API_KEY"),
        "mem0": _check_mem0,
    }


@router.get("/health/live")
def liveness() -> dict:
    return {"status": "alive"}


@router.get("/health/ready")
def readiness() -> dict:
    return build_readiness(dependency_checks())


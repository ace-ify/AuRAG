"""FastAPI dependency yielding a Neo4j session per request.

Provider imports stay lazy so API modules and pure service tests do not need
the full embedding/Qdrant stack merely to import their route definitions.
"""
import logging

logger = logging.getLogger(__name__)


class FallbackNeo4jSession:
    """Empty, honest fallback used only when Neo4j is unreachable.

    It returns NO rows — it never fabricates plant records (work orders,
    equipment, clauses). Fabricating domain data on failure is the single most
    dangerous thing this system could do: it would present invented
    maintenance/compliance facts as authoritative. Callers detect the outage
    via ResilientNeo4jSession.degraded and MUST abstain, rather than treat an
    empty result as an authoritative "nothing found".
    """

    def run(self, query: str, **kwargs):
        class FallbackResult:
            def data(self):
                return []

            def single(self):
                return None

            def values(self, *keys):
                return []

            def __iter__(self):
                return iter(())

        return FallbackResult()


class ResilientResult:
    """Wraps a live Neo4j result. If reading it raises, we log, mark the
    session degraded, and return EMPTY — never fabricated data."""

    def __init__(self, real_result, fallback_result, on_fallback=None):
        self._real = real_result
        self._fallback = fallback_result
        self._on_fallback = on_fallback

    def _degrade(self, method: str, exc: Exception) -> None:
        logger.warning("Neo4j result.%s failed (%s); degraded, returning empty.", method, exc)
        if self._on_fallback is not None:
            self._on_fallback()

    def data(self):
        try:
            return self._real.data()
        except Exception as exc:
            self._degrade("data()", exc)
            return self._fallback.data()

    def single(self):
        try:
            return self._real.single()
        except Exception as exc:
            self._degrade("single()", exc)
            return self._fallback.single()

    def values(self, *keys):
        try:
            return self._real.values(*keys)
        except Exception as exc:
            self._degrade("values()", exc)
            return self._fallback.values(*keys)

    def __iter__(self):
        try:
            return iter(self._real)
        except Exception as exc:
            self._degrade("__iter__()", exc)
            return iter(self._fallback)


class ResilientNeo4jSession:
    """Wraps a Neo4j session so a graph outage degrades to EMPTY results
    (never fabricated ones) and is observable via `degraded`."""

    def __init__(self, real_session=None):
        self._real_session = real_session
        self._fallback = FallbackNeo4jSession()
        # No live session at all -> already degraded.
        self.degraded = real_session is None

    def _mark_degraded(self) -> None:
        self.degraded = True

    def run(self, query: str, **kwargs):
        if self._real_session is not None:
            try:
                real_res = self._real_session.run(query, **kwargs)
                return ResilientResult(real_res, self._fallback.run(query, **kwargs), self._mark_degraded)
            except Exception as exc:
                logger.warning("Live Neo4j run failed (%s); degraded, returning empty.", exc)
                self._mark_degraded()
        return self._fallback.run(query, **kwargs)

    def close(self):
        if self._real_session is not None:
            try:
                self._real_session.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _get_neo4j_database() -> str | None:
    """Replicate retrieval.index_chunks.get_database() without importing it."""
    import os
    db = os.environ.get("NEO4J_DATABASE")
    if not db or db in ("neo4j", "None", ""):
        user = os.environ.get("NEO4J_USERNAME")
        if user and user != "neo4j":
            return user
        return None
    return db


_neo4j_driver = None


def _get_neo4j_driver():
    """Create Neo4j driver directly — avoids importing retrieval.index_chunks
    which triggers a 30-50s sentence_transformers model load."""
    global _neo4j_driver
    if _neo4j_driver is None:
        import os
        try:
            import truststore
            truststore.inject_into_ssl()
        except Exception:
            pass
        from neo4j import GraphDatabase
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USERNAME")
        pwd = os.environ.get("NEO4J_PASSWORD")
        if not all([uri, user, pwd]):
            raise RuntimeError("Missing NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD")
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=5.0, max_connection_lifetime=300)
    return _neo4j_driver


def get_session():
    real_session = None
    try:
        driver = _get_neo4j_driver()
        db = _get_neo4j_database()
        if db:
            real_session = driver.session(database=db)
        else:
            real_session = driver.session()
    except Exception as exc:
        logger.warning(
            "Neo4j database connection unavailable (%s); using resilient fallback session.",
            exc,
        )

    resilient = ResilientNeo4jSession(real_session)
    try:
        yield resilient
    finally:
        resilient.close()



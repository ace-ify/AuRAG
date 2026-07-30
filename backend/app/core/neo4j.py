"""FastAPI dependency yielding a Neo4j session per request.

Provider imports stay lazy so API modules and pure service tests do not need
the full embedding/Qdrant stack merely to import their route definitions.
"""


def get_session():
    from retrieval.index_chunks import get_database, get_driver

    driver = get_driver()
    with driver.session(database=get_database()) as session:
        yield session

"""Graph-traversal retrieval: closed-world entity extraction from the query
text (reusing ingestion.pipeline's fuzzy matchers — no new matching logic,
no LLM call per query), then traversal to both linked Chunks and structured
records (FailureEvent/WorkOrder/RegulatoryClause/Procedure), synthesized
into passage text on the fly since those records don't have a .text field.

depth=1 is the original direct-neighbor lookup. depth>=2 additionally walks
the HAS_PART assembly graph (both directions) and pulls evidence attached to
an equipment's parent assembly and its sub-components — genuinely connected
facts that neither dense/BM25 (they aren't in the text corpus) nor a 1-hop
lookup can surface. Example: a compliance question about vessel V-301 reaches
the relief valve PSV-701 (V-301 HAS_PART PSV-701) and its missed-calibration
clause, which no plain-RAG retriever could connect.
"""
import re

from ingestion.pipeline import load_known_entities, match_equipment, match_person

_PUNCT_RE = re.compile(r"[?.,;:!]+$")


def extract_query_entities(query: str, known_tags: set[str], known_names: set[str]) -> tuple[list[str], list[str]]:
    """Returns (matched_tags, matched_names). Equipment: each whitespace token
    tried as-is. Person: bigrams tried, since seed names are all "First Last"."""
    words = [_PUNCT_RE.sub("", w) for w in query.split()]

    tags = []
    for w in words:
        hit = match_equipment(w, known_tags)
        if hit and hit not in tags:
            tags.append(hit)

    names = []
    for a, b in zip(words, words[1:]):
        hit = match_person(f"{a} {b}", known_names)
        if hit and hit not in names:
            names.append(hit)

    return tags, names


def _fmt_failure(fe: dict, via: str = "") -> str:
    return f"{fe['id']} ({fe['date']}): {fe['symptom']} — root cause: {fe['root_cause']}{via}"


def _fmt_work_order(wo: dict, via: str = "") -> str:
    return f"{wo['id']} ({wo['date']}, {wo['type']}, {wo['status']}): {wo['description']}{via}"


def _fmt_clause(rc: dict, via: str = "") -> str:
    return f"{rc['id']} ({rc['source']}): {rc['text']}{via}"


def _fmt_procedure(proc: dict, via: str = "") -> str:
    return f"{proc['id']} v{proc['version']}: {proc['title']}{via}"


_EQUIPMENT_CYPHER = """
MATCH (e:Equipment {tag_id:$tag})
OPTIONAL MATCH (fe:FailureEvent)-[:OCCURRED_ON]->(e)
OPTIONAL MATCH (wo:WorkOrder)-[:PERFORMED_ON]->(e)
OPTIONAL MATCH (rc:RegulatoryClause)-[:APPLIES_TO]->(e)
OPTIONAL MATCH (proc:Procedure)-[:GOVERNS]->(e)
OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
RETURN
  collect(DISTINCT {id: fe.id, date: fe.date, symptom: fe.symptom, root_cause: fe.root_cause}) AS failure_events,
  collect(DISTINCT {id: wo.id, date: wo.date, type: wo.type, status: wo.status, description: wo.description}) AS work_orders,
  collect(DISTINCT {id: rc.clause_id, source: rc.source, text: rc.requirement_text}) AS clauses,
  collect(DISTINCT {id: proc.id, title: proc.title, version: proc.version}) AS procedures,
  collect(DISTINCT {id: c.id, text: c.text}) AS chunks
"""

# Second hop across the HAS_PART assembly graph (undirected), one row per
# connected neighbour so its evidence can be attributed to the traversed edge.
_EQUIPMENT_2HOP_CYPHER = """
MATCH (e:Equipment {tag_id:$tag})-[:HAS_PART]-(nbr:Equipment)
OPTIONAL MATCH (fe:FailureEvent)-[:OCCURRED_ON]->(nbr)
OPTIONAL MATCH (wo:WorkOrder)-[:PERFORMED_ON]->(nbr)
OPTIONAL MATCH (rc:RegulatoryClause)-[:APPLIES_TO]->(nbr)
OPTIONAL MATCH (proc:Procedure)-[:GOVERNS]->(nbr)
RETURN nbr.tag_id AS nbr_tag,
  collect(DISTINCT {id: fe.id, date: fe.date, symptom: fe.symptom, root_cause: fe.root_cause}) AS failure_events,
  collect(DISTINCT {id: wo.id, date: wo.date, type: wo.type, status: wo.status, description: wo.description}) AS work_orders,
  collect(DISTINCT {id: rc.clause_id, source: rc.source, text: rc.requirement_text}) AS clauses,
  collect(DISTINCT {id: proc.id, title: proc.title, version: proc.version}) AS procedures
"""

_PERSON_CYPHER = """
MATCH (p:Person {name:$name})
OPTIONAL MATCH (wo:WorkOrder)-[:PERFORMED_BY]->(p)
OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(p)
RETURN
  collect(DISTINCT {id: wo.id, date: wo.date, type: wo.type, status: wo.status, description: wo.description}) AS work_orders,
  collect(DISTINCT {id: c.id, text: c.text}) AS chunks
"""


def _add_equipment_1hop(session, tag: str, seen: dict[str, str]) -> None:
    row = session.run(_EQUIPMENT_CYPHER, tag=tag).single()
    if not row:
        return
    for fe in row["failure_events"]:
        if fe["id"]:
            seen.setdefault(fe["id"], _fmt_failure(fe))
    for wo in row["work_orders"]:
        if wo["id"]:
            seen.setdefault(wo["id"], _fmt_work_order(wo))
    for rc in row["clauses"]:
        if rc["id"]:
            seen.setdefault(rc["id"], _fmt_clause(rc))
    for proc in row["procedures"]:
        if proc["id"]:
            seen.setdefault(proc["id"], _fmt_procedure(proc))
    for c in row["chunks"]:
        if c["id"]:
            seen.setdefault(c["id"], c["text"])


def _add_equipment_2hop(session, tag: str, seen: dict[str, str]) -> None:
    """HAS_PART assembly neighbours. Records already surfaced 1-hop keep their
    (cleaner) text — setdefault means a 2-hop record only lands if the direct
    hop didn't already carry it."""
    for row in session.run(_EQUIPMENT_2HOP_CYPHER, tag=tag).data():
        nbr = row.get("nbr_tag")
        via = f" [related to {tag} via HAS_PART→{nbr}]" if nbr else ""
        for fe in row["failure_events"]:
            if fe["id"]:
                seen.setdefault(fe["id"], _fmt_failure(fe, via))
        for wo in row["work_orders"]:
            if wo["id"]:
                seen.setdefault(wo["id"], _fmt_work_order(wo, via))
        for rc in row["clauses"]:
            if rc["id"]:
                seen.setdefault(rc["id"], _fmt_clause(rc, via))
        for proc in row["procedures"]:
            if proc["id"]:
                seen.setdefault(proc["id"], _fmt_procedure(proc, via))


def traverse(session, query: str, top_k: int = 5, depth: int = 1) -> list[tuple[str, str]]:
    """Returns [(key, text), ...] candidate passages, deduped by key.

    depth=1 (default): direct neighbours only — preserves the tuned behaviour
    RCA/Compliance/Lessons-Learned rely on. depth>=2: also walk the HAS_PART
    assembly graph so connected-equipment evidence surfaces (used by the
    Copilot hybrid path)."""
    known_tags, known_names = load_known_entities(session)
    tags, names = extract_query_entities(query, known_tags, known_names)

    seen: dict[str, str] = {}

    for tag in tags:
        _add_equipment_1hop(session, tag, seen)
    if depth >= 2:
        for tag in tags:
            _add_equipment_2hop(session, tag, seen)

    for name in names:
        row = session.run(_PERSON_CYPHER, name=name).single()
        if not row:
            continue
        for wo in row["work_orders"]:
            if wo["id"]:
                seen.setdefault(wo["id"], _fmt_work_order(wo))
        for c in row["chunks"]:
            if c["id"]:
                seen.setdefault(c["id"], c["text"])

    return list(seen.items())[:top_k] if top_k else list(seen.items())


if __name__ == "__main__":
    import truststore
    truststore.inject_into_ssl()
    from retrieval.index_chunks import get_driver, get_database

    driver, db = get_driver(), get_database()
    with driver.session(database=db) as session:
        results = traverse(session, "Why did P-101 fail in March 2025?")
    assert results, "expected graph-traversal hits for a P-101 query"
    assert any(k.startswith("FE-") for k, _ in results), f"expected a FailureEvent hit, got {[k for k,_ in results]}"
    for key, text in results:
        print(f"{key}: {text[:80]!r}")

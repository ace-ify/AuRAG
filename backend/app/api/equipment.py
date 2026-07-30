"""GET /api/equipment — small new Cypher (not a reuse gap:
ingestion.pipeline.load_known_entities() only returns a bare tag_id set, no
name/type, so this is genuinely new but genuinely small). Feeds the
telemetry panel's equipment selector."""
from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.neo4j import get_session

router = APIRouter()


@router.get("/equipment")
def list_equipment(session=Depends(get_session)) -> list[dict]:
    try:
        rows = session.run("MATCH (e:Equipment) RETURN e.tag_id AS tag_id, e.name AS name, e.type AS type ORDER BY e.tag_id").data()
        return [{"tag_id": r["tag_id"], "name": r["name"], "type": r["type"]} for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "equipment_failed", "detail": str(exc)}) from exc

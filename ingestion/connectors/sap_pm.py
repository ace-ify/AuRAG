"""SAP Plant Maintenance (PM) / EAM connector.

Two modes, selected by `mode` (or the CONNECTOR_MODE env var; default "mock"):

- **live**: genuine OData v4 network I/O against a configured SAP Gateway / S4
  OData service (SAP_ODATA_URL), with real pagination, auth, health probing,
  and error handling via ingestion.connectors.odata.ODataClient. Field mapping
  is configurable because entity/field names differ across SAP releases.
- **mock** (default): serves the in-repo fixtures below for offline demos and
  tests. These are SIMULATED records, not a live SAP feed — health_check()
  returns True only because there is nothing to reach.

This is the one connector wired end-to-end to real network I/O; the OSIsoft PI,
SharePoint, and QMS connectors remain simulation fixtures (see their modules).
"""
import os
from typing import Any

from ingestion.connectors.base import BaseConnector, SyncResult
from ingestion.connectors.odata import ODataClient, ODataError
from ingestion.tag_normalizer import normalize_equipment_tag


# --- Simulated fixtures (mock mode only) ------------------------------------
MOCK_SAP_EQUIPMENT = [
    {"sap_id": "EQ-1001", "raw_tag": "P101", "description": "Boiler Feed Water Pump A",
     "floc": "MUM-U1-BFW-P101", "manufacturer": "Sulzer", "model": "MSD-2",
     "criticality": "HIGH", "status": "ACTIVE"},
    {"sap_id": "EQ-1002", "raw_tag": "P_101_B", "description": "Boiler Feed Water Pump B (Standby)",
     "floc": "MUM-U1-BFW-P101B", "manufacturer": "Sulzer", "model": "MSD-2",
     "criticality": "HIGH", "status": "STANDBY"},
    {"sap_id": "EQ-1003", "raw_tag": "TK-301", "description": "Condensate Storage Tank",
     "floc": "MUM-U1-COND-TK301", "manufacturer": "L&T Heavy Engineering",
     "model": "Atmospheric-500m3", "criticality": "MEDIUM", "status": "ACTIVE"},
]

MOCK_SAP_WORK_ORDERS = [
    {"order_id": "4000101", "raw_tag": "P101", "order_type": "PM02",
     "short_text": "Drive-end bearing replacement due to thermal spike",
     "priority": "1-VERY HIGH", "status": "TECO", "created_on": "2025-06-12"},
    {"order_id": "4000102", "raw_tag": "TK-301", "order_type": "PM01",
     "short_text": "Annual ultrasonic wall thickness inspection",
     "priority": "3-MEDIUM", "status": "TECO", "created_on": "2025-07-20"},
]

# Default OData field mapping (override per SAP release via constructor/env).
_EQUIP_FIELDS = {"sap_id": "Equipment", "raw_tag": "TagId", "description": "Description",
                 "floc": "FunctionalLocation", "criticality": "ABCIndicator"}
_WO_FIELDS = {"order_id": "OrderId", "raw_tag": "TagId", "order_type": "OrderType",
              "short_text": "ShortText", "status": "SystemStatus", "created_on": "CreatedOn"}


class SAPPMConnector(BaseConnector):
    """SAP PM/EAM connector — live OData or simulated fixtures (see module doc)."""

    def __init__(
        self,
        connector_id: str,
        site_id: str = "plant-mumbai-01",
        *,
        mode: str | None = None,
        odata_url: str | None = None,
        equipment_set: str | None = None,
        work_order_set: str | None = None,
        auth_header: str | None = None,
        transport: Any = None,
    ):
        super().__init__(connector_id, site_id)
        self.mode = (mode or os.environ.get("CONNECTOR_MODE", "mock")).lower()
        self.odata_url = odata_url or os.environ.get("SAP_ODATA_URL")
        self.equipment_set = equipment_set or os.environ.get("SAP_ODATA_EQUIPMENT_SET", "EquipmentSet")
        self.work_order_set = work_order_set or os.environ.get("SAP_ODATA_WORKORDER_SET", "MaintenanceOrderSet")
        self.auth_header = auth_header or os.environ.get("SAP_ODATA_AUTH_HEADER")
        self._transport = transport  # injected in tests (httpx.MockTransport)

    @property
    def source_type(self) -> str:
        return "SAP_PM"

    def _client(self) -> ODataClient:
        if not self.odata_url:
            raise ODataError("live mode requires SAP_ODATA_URL")
        return ODataClient(self.odata_url, auth_header=self.auth_header, transport=self._transport)

    def health_check(self) -> bool:
        """Mock mode is trivially reachable; live mode does a real OData probe."""
        if self.mode != "live":
            return True
        try:
            with self._client() as client:
                return client.probe(self.equipment_set)
        except ODataError:
            return False

    def sync(self, cursor: str | None = None, limit: int = 100) -> SyncResult:
        if self.mode == "live":
            return self._sync_live(cursor, limit)
        return self._sync_fixtures(limit)

    # --- live path ----------------------------------------------------------
    def _sync_live(self, cursor: str | None, limit: int) -> SyncResult:
        params = {"$skiptoken": cursor} if cursor else None
        try:
            with self._client() as client:
                equipment_rows = client.fetch_all(self.equipment_set, page_size=min(limit, 100),
                                                   max_records=limit, params=params)
                wo_rows = client.fetch_all(self.work_order_set, page_size=min(limit, 100),
                                           max_records=limit, params=params)
        except ODataError as exc:
            # Fail honestly — do NOT fall back to fixtures and present them as live.
            return SyncResult(connector_id=self.connector_id, source_type=self.source_type,
                              records_synced=0, records=[], success=False, error_message=str(exc))

        records: list[dict[str, Any]] = []
        for row in equipment_rows:
            records.append(self._equip_record(
                sap_id=row.get(_EQUIP_FIELDS["sap_id"]),
                raw_tag=row.get(_EQUIP_FIELDS["raw_tag"]) or row.get(_EQUIP_FIELDS["sap_id"], ""),
                description=row.get(_EQUIP_FIELDS["description"], ""),
                floc=row.get(_EQUIP_FIELDS["floc"], ""),
                criticality=row.get(_EQUIP_FIELDS["criticality"], ""),
            ))
        for row in wo_rows:
            records.append(self._wo_record(
                order_id=row.get(_WO_FIELDS["order_id"]),
                raw_tag=row.get(_WO_FIELDS["raw_tag"], ""),
                order_type=row.get(_WO_FIELDS["order_type"], ""),
                short_text=row.get(_WO_FIELDS["short_text"], ""),
                status=row.get(_WO_FIELDS["status"], ""),
                created_on=row.get(_WO_FIELDS["created_on"], ""),
            ))

        return SyncResult(connector_id=self.connector_id, source_type=self.source_type,
                          records_synced=len(records), records=records, success=True,
                          new_cursor=f"skip_{len(records)}")

    # --- mock path ----------------------------------------------------------
    def _sync_fixtures(self, limit: int) -> SyncResult:
        records: list[dict[str, Any]] = []
        for eq in MOCK_SAP_EQUIPMENT[:limit]:
            records.append(self._equip_record(sap_id=eq["sap_id"], raw_tag=eq["raw_tag"],
                                               description=eq["description"], floc=eq["floc"],
                                               criticality=eq["criticality"]))
        for wo in MOCK_SAP_WORK_ORDERS[:limit]:
            records.append(self._wo_record(order_id=wo["order_id"], raw_tag=wo["raw_tag"],
                                           order_type=wo["order_type"], short_text=wo["short_text"],
                                           status=wo["status"], created_on=wo["created_on"]))
        return SyncResult(connector_id=self.connector_id, source_type=self.source_type,
                          records_synced=len(records), records=records, success=True,
                          new_cursor=f"mock_cursor_{len(records)}")

    # --- shared record shaping ---------------------------------------------
    def _equip_record(self, *, sap_id, raw_tag, description, floc, criticality) -> dict[str, Any]:
        return {
            "entity_type": "Equipment",
            "canonical_tag": normalize_equipment_tag(raw_tag) or raw_tag,
            "raw_tag": raw_tag,
            "sap_id": sap_id,
            "description": description,
            "functional_location": floc,
            "criticality": criticality,
            "site_id": self.site_id,
        }

    def _wo_record(self, *, order_id, raw_tag, order_type, short_text, status, created_on) -> dict[str, Any]:
        return {
            "entity_type": "WorkOrder",
            "order_id": order_id,
            "canonical_tag": normalize_equipment_tag(raw_tag) or raw_tag,
            "order_type": order_type,
            "description": short_text,
            "status": status,
            "created_on": created_on,
            "site_id": self.site_id,
        }

"""SAP PM connector — live OData path proven with httpx.MockTransport (real
request construction, pagination via @odata.nextLink, health probe, and honest
failure on HTTP error), plus the default mock path staying intact."""
import httpx

from ingestion.connectors.odata import ODataClient, ODataError
from ingestion.connectors.sap_pm import SAPPMConnector

_BASE = "https://sap.example/odata"


def _ok_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = request.url.params
    if path.endswith("/EquipmentSet"):
        if "$skip" in params:  # page 2 (followed via @odata.nextLink)
            return httpx.Response(200, json={"value": [
                {"Equipment": "EQ-9002", "TagId": "C-201", "Description": "Live Compressor",
                 "FunctionalLocation": "FL-2", "ABCIndicator": "HIGH"},
            ]})
        return httpx.Response(200, json={
            "value": [{"Equipment": "EQ-9001", "TagId": "P901", "Description": "Live Pump",
                       "FunctionalLocation": "FL-1", "ABCIndicator": "HIGH"}],
            "@odata.nextLink": f"{_BASE}/EquipmentSet?$skip=1",
        })
    if path.endswith("/MaintenanceOrderSet"):
        return httpx.Response(200, json={"value": [
            {"OrderId": "WO-9001", "TagId": "P901", "OrderType": "PM01",
             "ShortText": "Live work order", "SystemStatus": "OPEN", "CreatedOn": "2026-01-01"},
        ]})
    return httpx.Response(404, json={"error": "not found"})


def _live_connector(handler):
    return SAPPMConnector(
        connector_id="sap_live_test", mode="live", odata_url=_BASE,
        transport=httpx.MockTransport(handler),
    )


def test_live_health_check_probes_real_endpoint():
    assert _live_connector(_ok_handler).health_check() is True


def test_live_sync_paginates_and_maps_fields():
    result = _live_connector(_ok_handler).sync(limit=10)

    assert result.success is True
    equipment = [r for r in result.records if r["entity_type"] == "Equipment"]
    work_orders = [r for r in result.records if r["entity_type"] == "WorkOrder"]

    # Two equipment records means @odata.nextLink pagination was followed.
    assert {e["sap_id"] for e in equipment} == {"EQ-9001", "EQ-9002"}
    assert any(e["raw_tag"] == "P901" and e["canonical_tag"] for e in equipment)
    assert work_orders and work_orders[0]["order_id"] == "WO-9001"


def test_live_sync_fails_honestly_on_http_error():
    def boom(_request):
        return httpx.Response(503, json={"error": "gateway down"})

    result = _live_connector(boom).sync(limit=10)
    assert result.success is False
    assert result.records == []
    assert "503" in (result.error_message or "")


def test_live_health_check_false_on_error():
    def boom(_request):
        return httpx.Response(401, json={"error": "unauthorized"})

    assert _live_connector(boom).health_check() is False


def test_live_mode_without_url_fails_closed():
    connector = SAPPMConnector(connector_id="no_url", mode="live", odata_url=None)
    assert connector.health_check() is False
    assert connector.sync().success is False


def test_mock_mode_is_default_and_unchanged():
    connector = SAPPMConnector(connector_id="sap_mock")
    assert connector.mode == "mock"
    assert connector.health_check() is True
    result = connector.sync(limit=10)
    assert result.success is True
    assert any(r.get("sap_id") == "EQ-1001" and r["canonical_tag"] == "P-101" for r in result.records)


def test_odata_client_stops_at_max_records():
    def infinite(request):
        skip = request.url.params.get("$skip", "0")
        return httpx.Response(200, json={
            "value": [{"id": skip}],
            "@odata.nextLink": f"{_BASE}/Thing?$skip={int(skip) + 1}",
        })

    with ODataClient(_BASE, transport=httpx.MockTransport(infinite)) as client:
        rows = client.fetch_all("Thing", page_size=1, max_records=3)
    assert len(rows) == 3  # bounded despite an endless nextLink chain


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

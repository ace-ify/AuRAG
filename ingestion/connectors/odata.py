"""Minimal real OData v4 client — genuine network I/O, pagination, and error
handling. This is the piece that proves the enterprise-connector abstraction
against a live endpoint (a real SAP Gateway / S4 OData service, or any OData v4
source), as opposed to the in-repo fixtures used for offline demos.

Kept dependency-light (httpx only) and transport-injectable so it is unit
tested deterministically with httpx.MockTransport — no network in CI."""
from __future__ import annotations

import httpx


class ODataError(RuntimeError):
    """Raised on a non-2xx OData response or transport failure."""


class ODataClient:
    def __init__(
        self,
        base_url: str,
        *,
        auth_header: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {"Accept": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def _get(self, url: str, params: dict | None = None) -> dict:
        try:
            resp = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise ODataError(f"transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise ODataError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ODataError(f"invalid JSON from OData service: {exc}") from exc

    def probe(self, entity_set: str) -> bool:
        """Real connectivity + auth check: fetch a single record. Any non-2xx
        (401/403 bad creds, 404 wrong set, 5xx down) returns False."""
        try:
            self._get(f"/{entity_set}", params={"$top": 1})
            return True
        except ODataError:
            return False

    def fetch_all(
        self,
        entity_set: str,
        *,
        page_size: int = 100,
        max_records: int = 1000,
        params: dict | None = None,
    ) -> list[dict]:
        """Fetch entities, following @odata.nextLink pagination until exhausted
        or max_records reached. The server-driven nextLink is opaque and carries
        its own query string, so we stop sending params once we start following
        it."""
        records: list[dict] = []
        url: str | None = f"/{entity_set}"
        query: dict | None = {"$top": page_size, **(params or {})}

        while url and len(records) < max_records:
            data = self._get(url, params=query)
            records.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
            query = None  # nextLink already encodes $skip/$top

        return records[:max_records]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ODataClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

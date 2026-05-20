"""Base class for Kndc parsers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class KndcBulletinBase:
    """Base class for Kndc parsers."""

    LIST_URL = "https://kndc.kz/kndc/pagecontent/alarm-bulletin/getOriginList.php"

    def __init__(
        self,
        output_dir: str = "data/kndc",
        limit: int = 25,
        timeout: int = 15,
    ) -> None:
        """Parser initialization."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        self.timeout = int(timeout)
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json, */*"})

    def _fetch_listing(
        self, desc: bool = False, start: int = 0
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str | int] = {
            "orderby": "epochtime",
            "desc": "yes" if desc else "no",
            "start": int(start),
            "limit": int(self.limit),
        }
        try:
            resp = self._session.get(self.LIST_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = self._parse_html_wrapped_json(resp.text)
            if isinstance(data, list):
                return [
                    cast(Dict[str, Any], row) for row in data if isinstance(row, dict)
                ]
            if isinstance(data, dict):
                items_obj: object = data.get("items", [])
                if isinstance(items_obj, list):
                    return [
                        cast(Dict[str, Any], row)
                        for row in items_obj
                        if isinstance(row, dict)
                    ]
            return []
        except Exception:
            logger.exception("Failed to fetch bulletin listing")
            return []

    def _parse_html_wrapped_json(self, payload: str) -> object:
        soup = BeautifulSoup(payload or "", "html.parser")
        body = (
            soup.body.get_text(strip=True) if soup.body else soup.get_text(strip=True)
        )
        if not body:
            return []
        parsed: object = json.loads(body)
        return parsed

    def _normalize(self, event: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        depth_km = float(event.get("depth", 0.0))
        parsed = {
            "event_id": int(event.get("id", 0)),
            "epoch_time": int(event.get("epochtime", 0)),
            "occurred_at_utc": None,
            "latitude": float(event.get("lat", 0.0)),
            "longitude": float(event.get("lon", 0.0)),
            "depth_km": depth_km,
            "mb": float(event.get("mb", 0.0)) if event.get("mb") is not None else None,
            "mpv": float(event.get("mpv", 0.0))
            if event.get("mpv") is not None
            else None,
            "energy_class_k": float(event.get("class", 0.0))
            if event.get("class") is not None
            else None,
            "geographic_region": event.get("gregion"),
            "seismic_region": event.get("sregion"),
            "quality": event.get("qual"),
            "author": event.get("auth"),
            "last_updated": event.get("lddate"),
            "raw": event,
        }

        try:
            evdate = event.get("evdate")
            evtime = event.get("evtime")
            if evdate and evtime:
                parsed["occurred_at_utc"] = datetime.strptime(
                    f"{evdate} {evtime}", "%Y-%m-%d %H:%M:%S"
                )
        except Exception:
            parsed["occurred_at_utc"] = None

        depth = depth_km
        if depth < 70:
            depth_description = "shallow earthquake"
        elif depth < 300:
            depth_description = "intermediate-depth earthquake"
        else:
            depth_description = "deep earthquake"

        enriched_text = f"""Official earthquake bulletin from KNDC. Event ID: {parsed['event_id']}, Date and time (UTC): {parsed['occurred_at_utc']}, Geographic region: {parsed['geographic_region']}, Seismic region: {parsed['seismic_region']}. Coordinates: Latitude {parsed['latitude']}, Longitude {parsed['longitude']}, Depth: {parsed['depth_km']} km. This was a {depth_description}. Magnitude scales: mb = {parsed['mb']}, mpv = {parsed['mpv']}. Energy class K: {parsed['energy_class_k']}. Source author:{parsed['author']}. Quality classification: {parsed['quality']}. Last official update: {parsed['last_updated']}""".strip()  # noqa: E501

        return parsed, enriched_text

    def _latest_snapshot_for_event(self, event_id: int) -> Optional[Path]:
        matches = sorted(self.output_dir.glob(f"kndc_bulletin_{int(event_id)}_*.json"))
        if not matches:
            return None
        return matches[-1]

    def _snapshot_exists(self, event_id: int) -> bool:
        return self._latest_snapshot_for_event(event_id) is not None

    def _load_snapshot_parsed(self, event_id: int) -> Optional[Dict[str, Any]]:
        p = self._latest_snapshot_for_event(event_id)
        if not p:
            return None
        try:
            with p.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
                return payload.get("parsed") if isinstance(payload, dict) else None
        except Exception:
            return None

    def _is_parsed_incomplete(
        self, parsed: Dict[str, Any], event: Dict[str, Any]
    ) -> bool:
        if not parsed:
            return True
        if int(parsed.get("event_id", 0)) == 0:
            return True
        if parsed.get("latitude") is None or parsed.get("longitude") is None:
            return True
        lat = float(parsed.get("latitude") or 0.0)
        lon = float(parsed.get("longitude") or 0.0)
        if lat == 0.0 and lon == 0.0:
            return True
        if int(parsed.get("epoch_time", 0)) == 0:
            return True
        if parsed.get("occurred_at_utc") is None:
            # try to see if event provides a valid evdate/evtime
            if not (event.get("evdate") and event.get("evtime")):
                return True
        return False

    def _save_snapshot(
        self,
        event_id: int,
        parsed: Dict[str, Any],
        enriched_text: str,
        force: bool = False,
    ) -> Path:
        """Save snapshot. If force is False, skip when a snapshot already exists.

        If force is True, always write a new snapshot file (does not delete old ones).
        """
        if not force and self._snapshot_exists(event_id):
            logger.info("Skipping duplicate bulletin event_id=%d", event_id)
            existing = self._latest_snapshot_for_event(event_id)
            assert existing is not None
            return existing

        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"kndc_bulletin_{event_id}_{ts}.json"
        out = self.output_dir / name
        payload = {
            "parsed": parsed,
            "enriched_text": enriched_text,
            "raw": parsed.get("raw"),
        }
        with out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        return out

    def _bootstrap_next_id_from_snapshots(self) -> Optional[int]:
        max_event_id: Optional[int] = None
        for path in self.output_dir.glob("kndc_bulletin_*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                parsed = payload.get("parsed", {}) if isinstance(payload, dict) else {}
                event_id = int(parsed.get("event_id", 0))
                if event_id > 0 and (max_event_id is None or event_id > max_event_id):
                    max_event_id = event_id
            except Exception:
                continue

        if max_event_id is None:
            return None

        return max_event_id + 1

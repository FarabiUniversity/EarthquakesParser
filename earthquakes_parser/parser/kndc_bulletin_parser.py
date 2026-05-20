"""Hourly KNDC bulletin listing parser.

Fetches KNDC bulletin listing rows and persists per-event snapshots.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class KndcBulletinHourlyParser:
    """Hourly parser for the KNDC bulletin listing.

    Fetches the latest listing (descending epochtime) and writes new bulletins
    as snapshots with `parsed` and `enriched_text`. It keeps a cursor file for
    new-record discovery, mirroring the `kndc_parser` flow.
    """

    LIST_URL = "https://kndc.kz/kndc/pagecontent/alarm-bulletin/getOriginList.php"

    def __init__(
        self,
        output_dir: str = "data/kndc",
        limit: int = 25,
        timeout: int = 15,
    ) -> None:
        """Initialize the hourly bulletin parser."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        self.timeout = int(timeout)
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json, */*"})
        self._next_id_file = self.output_dir / "bulletin_next_id.txt"

    def _fetch_listing(self) -> List[Dict[str, Any]]:
        params: Dict[str, str | int] = {
            "orderby": "epochtime",
            "desc": "yes",
            "start": 0,
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

        enriched_text = f"""
Official earthquake bulletin from KNDC.

Event ID: {parsed['event_id']}

Date and time (UTC):
{parsed['occurred_at_utc']}

Geographic region:
{parsed['geographic_region']}

Seismic region:
{parsed['seismic_region']}

Coordinates:
Latitude {parsed['latitude']}
Longitude {parsed['longitude']}

Depth:
{parsed['depth_km']} km

This was an {depth_description}.

Magnitude scales:
mb = {parsed['mb']}
mpv = {parsed['mpv']}

Energy class K:
{parsed['energy_class_k']}

Source author:
{parsed['author']}

Quality classification:
{parsed['quality']}

Last official update:
{parsed['last_updated']}
""".strip()

        return parsed, enriched_text

    def _load_next_id(self) -> int:
        if not self._next_id_file.exists():
            bootstrapped = self._bootstrap_next_id_from_snapshots()
            if bootstrapped is not None:
                self._save_next_id(bootstrapped)
                logger.info(
                    "Bootstrapped bulletin next_id=%d from existing snapshots",
                    bootstrapped,
                )
                return bootstrapped
            return 1
        try:
            value = self._next_id_file.read_text(encoding="utf-8").strip()
            return int(value) if value else 1
        except Exception:
            return 1

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

    def _save_next_id(self, value: int) -> None:
        self._next_id_file.write_text(str(int(value)), encoding="utf-8")

    def _save_snapshot(
        self, event_id: int, parsed: Dict[str, Any], enriched_text: str
    ) -> Path:
        if self._snapshot_exists(event_id):
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

    def _snapshot_exists(self, event_id: int) -> bool:
        return self._latest_snapshot_for_event(event_id) is not None

    def _latest_snapshot_for_event(self, event_id: int) -> Optional[Path]:
        matches = sorted(self.output_dir.glob(f"kndc_bulletin_{int(event_id)}_*.json"))
        if not matches:
            return None
        return matches[-1]

    def run_once(self, max_per_run: int = 25) -> int:
        """Fetch the latest bulletin listing once and save new snapshots."""
        logger.info("Starting bulletin probe")
        listing = self._fetch_listing()
        if not listing:
            logger.info("Bulletin probe returned no rows")
            return 0

        next_id = self._load_next_id()
        logger.info("Starting bulletin probe. next_id=%s", next_id)

        found: List[tuple[int, Dict[str, Any], str]] = []
        probes = 0

        for event in listing:
            if probes >= max_per_run:
                break
            probes += 1

            try:
                eid = int(event.get("id", 0))
            except Exception:
                continue

            if eid < next_id:
                continue
            if self._snapshot_exists(eid):
                logger.info("Skipping duplicate bulletin event_id=%d", eid)
                continue

            parsed, enriched = self._normalize(event)
            found.append((eid, parsed, enriched))

        if not found:
            logger.info("No new bulletin ids found. Keeping next_id=%s", next_id)
            return 0

        new_ids = [eid for eid, _, _ in found]
        logger.info("Discovered %d new bulletin id(s): %s", len(new_ids), new_ids)

        for eid, parsed, enriched in found:
            p = self._save_snapshot(eid, parsed, enriched)
            logger.info("Saved bulletin snapshot %s", p)

        self._save_next_id(max(new_ids) + 1)
        logger.info("Advanced bulletin next_id to %d", max(new_ids) + 1)
        return len(found)

    def run(self, max_per_run: int = 25) -> int:
        """Compatibility wrapper around :meth:`run_once`."""
        return self.run_once(max_per_run=max_per_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kp = KndcBulletinHourlyParser()
    n = kp.run_once()
    print(f"Saved {n} new bulletin(s)")

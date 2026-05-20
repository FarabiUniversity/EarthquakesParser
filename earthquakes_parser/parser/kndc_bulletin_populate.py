"""Backfill KNDC bulletin listing into local snapshot files."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# python -m earthquakes_parser.parser.kndc_bulletin_populate
class KndcBulletinPopulator:
    """Backfill KNDC bulletin listing into local snapshots.

    This class fetches the bulletin listing (up to `limit`) and writes per-event
    snapshots containing `parsed` and `enriched_text` into `output_dir`.
    It is resumable via a small progress file.
    """

    LIST_URL = "https://kndc.kz/kndc/pagecontent/alarm-bulletin/getOriginList.php"

    def __init__(
        self,
        output_dir: str = "data/kndc",
        limit: int = 50000,
        timeout: int = 15,
    ) -> None:
        """Initialize the bulletin listing backfill."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        self.timeout = int(timeout)
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json, */*"})
        self._progress_file = self.output_dir / "bulletin_last_start.txt"

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

This was a {depth_description}.

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

    def _load_start_offset(self) -> int:
        if not self._progress_file.exists():
            return 0
        try:
            v = int(self._progress_file.read_text(encoding="utf-8").strip() or 0)
            return v
        except Exception:
            return 0

    def _save_start_offset(self, v: int) -> None:
        self._progress_file.write_text(str(int(v)), encoding="utf-8")

    def run_backfill(self, resume: bool = True) -> int:
        """Run a one-shot backfill. Returns number of processed events."""
        processed = 0
        start = self._load_start_offset() if resume else 0

        while True:
            listing = self._fetch_listing(desc=False, start=start)
            if not listing:
                break

            page_processed = 0
            # Listing is expected in ascending epochtime when desc=False
            for event in listing:
                try:
                    eid = int(event.get("id", 0))
                except Exception:
                    continue
                if self._snapshot_exists(eid):
                    logger.info("Skipping duplicate bulletin event_id=%d", eid)
                    continue

                parsed, enriched = self._normalize(event)
                p = self._save_snapshot(eid, parsed, enriched)
                logger.info("Saved bulletin snapshot %s", p)
                processed += 1
                page_processed += 1
                time.sleep(random.uniform(0.01, 0.1))

            start += len(listing)
            self._save_start_offset(start)

            if page_processed == 0:
                break
            if len(listing) < self.limit:
                break

        return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pop = KndcBulletinPopulator()
    n = pop.run_backfill()
    print(f"Processed {n} bulletin events")

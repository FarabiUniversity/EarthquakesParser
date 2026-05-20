"""Hourly KNDC bulletin listing parser.

Fetches KNDC bulletin listing rows and persists per-event snapshots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from earthquakes_parser.parser.kndc_bulletin_shared import LIST_URL as BULLETIN_LIST_URL
from earthquakes_parser.parser.kndc_bulletin_shared import (
    create_session,
    fetch_listing,
    latest_snapshot_for_event,
    normalize_event,
    save_snapshot,
    snapshot_exists,
)

logger = logging.getLogger(__name__)


class KndcBulletinHourlyParser:
    """Hourly parser for the KNDC bulletin listing.

    Fetches the latest listing (descending epochtime) and writes new bulletins
    as snapshots with `parsed` and `enriched_text`. It keeps a cursor file for
    new-record discovery, mirroring the `kndc_parser` flow.
    """

    LIST_URL = BULLETIN_LIST_URL

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
        self._session = create_session()
        self._next_id_file = self.output_dir / "bulletin_next_id.txt"

    def _fetch_listing(self) -> List[Dict[str, Any]]:
        return fetch_listing(
            self._session,
            limit=int(self.limit),
            timeout=int(self.timeout),
            desc=True,
            start=0,
        )

    def _normalize(self, event: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        return normalize_event(event)

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
        return save_snapshot(self.output_dir, event_id, parsed, enriched_text)

    def _snapshot_exists(self, event_id: int) -> bool:
        return snapshot_exists(self.output_dir, event_id)

    def _latest_snapshot_for_event(self, event_id: int) -> Optional[Path]:
        return latest_snapshot_for_event(self.output_dir, event_id)

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

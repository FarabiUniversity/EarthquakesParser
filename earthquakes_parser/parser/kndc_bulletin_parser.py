"""Hourly KNDC bulletin listing parser.

Fetches KNDC bulletin listing rows and ingests per-event snapshots into Supabase.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from earthquakes_parser.parser.kndc_bulletin_shared import LIST_URL as BULLETIN_LIST_URL
from earthquakes_parser.parser.kndc_bulletin_shared import (
    create_session,
    fetch_listing,
    normalize_event,
)
from earthquakes_parser.parser.kndc_supabase import KndcSupabaseStore
from earthquakes_parser.storage.supabase.database import SupabaseDB

logger = logging.getLogger(__name__)


class KndcBulletinHourlyParser:
    """Hourly parser for the KNDC bulletin listing.

    Fetches the latest listing (descending epochtime) and upserts new bulletins
    into Supabase with `parsed` and `enriched_text`.
    """

    LIST_URL = BULLETIN_LIST_URL

    def __init__(
        self,
        output_dir: str = "data/kndc",
        db: Optional[SupabaseDB] = None,
        limit: int = 25,
        timeout: int = 15,
    ) -> None:
        """Initialize the hourly bulletin parser."""
        self.output_dir = output_dir
        self.limit = int(limit)
        self.timeout = int(timeout)
        self._session = create_session()
        self.storage = KndcSupabaseStore(db=db)

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

    @staticmethod
    def _parse_event_id(value: Any) -> Optional[int]:
        try:
            eid = int(value)
        except (TypeError, ValueError):
            return None
        return eid if eid > 0 else None

    def _load_next_id(self) -> int:
        latest_event_id = self.storage.latest_event_id()
        return 1 if latest_event_id is None else latest_event_id + 1

    def _save_snapshot(
        self, event_id: int, parsed: Dict[str, Any], enriched_text: str
    ) -> Dict[str, Any]:
        return self.storage.upsert_bulletin(event_id, parsed, enriched_text)

    def _snapshot_exists(self, event_id: int) -> bool:
        return self.storage.bulletin_exists(event_id)

    def run_once(self, max_per_run: int = 25) -> int:
        """Fetch the latest bulletin listing once and upsert new rows."""
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

            eid = self._parse_event_id(event.get("id"))
            if eid is None:
                logger.debug(
                    "Skipping listing row with invalid id: %s", event.get("id")
                )
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
            stored = self._save_snapshot(eid, parsed, enriched)
            logger.info(
                "Upserted bulletin event_id=%s into Supabase", stored.get("event_id")
            )

        return len(found)

    def run(self, max_per_run: int = 25) -> int:
        """Compatibility wrapper around :meth:`run_once`."""
        return self.run_once(max_per_run=max_per_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kp = KndcBulletinHourlyParser()
    n = kp.run_once()
    print(f"Upserted {n} new bulletin(s)")

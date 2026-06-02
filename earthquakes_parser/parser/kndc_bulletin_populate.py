"""Backfill KNDC bulletin listing into local snapshot files."""

from __future__ import annotations

import logging
import secrets
import time
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


# python -m earthquakes_parser.parser.kndc_bulletin_populate
class KndcBulletinPopulator:
    """Backfill KNDC bulletin listing into local snapshots.

    This class fetches the bulletin listing and writes per-event snapshots
    containing `parsed` and `enriched_text` into `output_dir`.

    It intentionally scans the full listing on every run so it can fill gaps
    (e.g., if some local snapshot files were deleted).
    """

    LIST_URL = BULLETIN_LIST_URL

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
        self._session = create_session()

    def _fetch_listing(
        self, desc: bool = False, start: int = 0
    ) -> List[Dict[str, Any]]:
        return fetch_listing(
            self._session,
            limit=int(self.limit),
            timeout=int(self.timeout),
            desc=bool(desc),
            start=int(start),
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

    @staticmethod
    def _delay_seconds(min_delay: float, max_delay: float) -> float:
        if max_delay <= min_delay:
            return max(min_delay, 0.0)
        span = max_delay - min_delay
        jitter = secrets.randbelow(1_000_000) / 1_000_000
        return min_delay + (span * jitter)

    def _save_snapshot(
        self, event_id: int, parsed: Dict[str, Any], enriched_text: str
    ) -> Path:
        return save_snapshot(self.output_dir, event_id, parsed, enriched_text)

    def _snapshot_exists(self, event_id: int) -> bool:
        return snapshot_exists(self.output_dir, event_id)

    def _latest_snapshot_for_event(self, event_id: int) -> Optional[Path]:
        return latest_snapshot_for_event(self.output_dir, event_id)

    def run_backfill(self) -> int:
        """Run a one-shot backfill across the entire listing.

        Returns the number of newly written snapshot files.
        """
        processed = 0
        start = 0

        while True:
            listing = self._fetch_listing(desc=False, start=start)
            if not listing:
                break

            # Listing is expected in ascending epochtime when desc=False
            for event in listing:
                eid = self._parse_event_id(event.get("id"))
                if eid is None:
                    logger.debug(
                        "Skipping listing row with invalid id: %s", event.get("id")
                    )
                    continue
                if self._snapshot_exists(eid):
                    logger.info("Skipping duplicate bulletin event_id=%d", eid)
                    continue

                parsed, enriched = self._normalize(event)
                p = self._save_snapshot(eid, parsed, enriched)
                logger.info("Saved bulletin snapshot %s", p)
                processed += 1
                time.sleep(self._delay_seconds(0.01, 0.1))

            start += len(listing)
            if len(listing) < self.limit:
                break

        return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pop = KndcBulletinPopulator()
    n = pop.run_backfill()
    print(f"Processed {n} bulletin events")

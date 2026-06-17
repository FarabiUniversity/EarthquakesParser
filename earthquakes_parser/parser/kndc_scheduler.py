"""Scheduler for KNDC parser.

This scheduler runs forever until interrupted. It keeps a rolling 25-id window
starting at `next_id`, probes those exact KNDC `newsid` values on every run, and
only advances the window when at least one real record appears.

Usage:
python -m earthquakes_parser.parser.kndc_scheduler
python -m earthquakes_parser.parser.kndc_scheduler --interval 3600
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import List

from earthquakes_parser.parser.kndc_bulletin_parser import KndcBulletinHourlyParser
from earthquakes_parser.parser.kndc_parser import KndcParser, KndcRecord
from earthquakes_parser.parser.kndc_supabase import KndcSupabaseStore
from earthquakes_parser.storage.supabase.database import SupabaseDB

logger = logging.getLogger(__name__)


def _delay_seconds(min_delay: float, max_delay: float) -> float:
    if max_delay <= min_delay:
        return max(min_delay, 0.0)
    span = max_delay - min_delay
    jitter = secrets.randbelow(1_000_000) / 1_000_000
    return min_delay + (span * jitter)


def probe_new_ids(
    parser: KndcParser,
    start_id: int,
    max_per_run: int = 25,
) -> tuple[list[tuple[int, KndcRecord]], int, bool]:
    """Probe sequential IDs starting at `start_id`.

    Returns a tuple of:
    - discovered ids that produced non-empty payloads
    - the next id to probe on the next cycle
    - whether at least one record was found
    """
    found: List[tuple[int, KndcRecord]] = []
    nid = int(start_id)
    probes = 0

    while probes < max_per_run:
        probes += 1
        rec = parser.fetch_by_newsid(nid)
        if rec:
            found.append((nid, rec))
            logger.debug("Found newsid %d", nid)

        time.sleep(_delay_seconds(parser.min_delay, parser.max_delay))

        nid += 1

    if found:
        return found, max(item[0] for item in found) + 1, True

    return found, start_id, False


def run_scheduler(
    interval: int = 3600,
    output_dir: str = "data/kndc",
    max_per_run: int = 25,
):
    """Run the KNDC scheduler loop until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    db = SupabaseDB()
    storage = KndcSupabaseStore(db=db)
    kp = KndcParser(output_dir=output_dir, db=db)

    # hourly bulletin parser (runs each scheduler cycle; writes local snapshots)
    bp = KndcBulletinHourlyParser(output_dir=output_dir, db=db, limit=25)

    while True:
        try:
            next_id = storage.latest_newsid() or 1

            logger.info("Starting probe. next_id=%s", next_id)

            # Run hourly bulletin parser first to capture any new bulletins
            try:
                new_bulletins = bp.run_once()
                if new_bulletins:
                    logger.info(
                        "Hourly bulletin parser upserted %d bulletin(s)", new_bulletins
                    )
            except Exception:
                logger.exception("Bulletin parser error")

            discovered, next_id, advanced = probe_new_ids(
                kp, next_id, max_per_run=max_per_run
            )
            if discovered:
                new_ids = [nid for nid, _ in discovered]
                logger.info("Discovered %d new id(s): %s", len(new_ids), new_ids)
                records = []
                for _nid, rec in discovered:
                    records.append(rec)
                    stored = kp.save(rec)
                    logger.info(
                        "Upserted KNDC newsid=%s into Supabase", stored.get("newsid")
                    )
                logger.info("Upserted %d record(s)", len(records))
            elif not advanced:
                logger.info(
                    "No records in current 25id window; retrying the same ids next run"
                )
            else:
                logger.info("No new ids found this run.")

        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user. Exiting.")
            break
        except Exception:
            logger.exception("Unexpected error in scheduler loop")

        logger.info("Sleeping for %d seconds...", interval)
        time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KNDC scheduler")
    parser.add_argument(
        "--interval", type=int, default=3600, help="Seconds between runs"
    )
    parser.add_argument("--output", default="data/kndc", help="Output directory")
    parser.add_argument("--max", type=int, default=25, help="Max new items per run")
    args = parser.parse_args()

    run_scheduler(interval=args.interval, output_dir=args.output, max_per_run=args.max)

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
import random
import time
from pathlib import Path
from typing import List, Set

from earthquakes_parser.parser.kndc_bulletin_parser import KndcBulletinHourlyParser
from earthquakes_parser.parser.kndc_parser import KndcParser, KndcRecord

logger = logging.getLogger(__name__)


SEEN_FILE = Path("data/kndc/seen_ids.txt")
NEXT_ID_FILE = Path("data/kndc/next_id.txt")


def load_seen() -> Set[int]:
    """Load the set of already-seen `newsid` values from disk."""
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SEEN_FILE.exists():
        return set()
    try:
        with SEEN_FILE.open("r", encoding="utf-8") as fh:
            return {int(line.strip()) for line in fh if line.strip()}
    except Exception:
        return set()


def save_seen(seen: Set[int]) -> None:
    """Persist the set of seen ids to disk."""
    with SEEN_FILE.open("w", encoding="utf-8") as fh:
        for i in sorted(seen):
            fh.write(f"{i}\n")


def load_next_id() -> int:
    """Load the next `newsid` cursor from disk."""
    if not NEXT_ID_FILE.exists():
        return 1
    try:
        value = NEXT_ID_FILE.read_text(encoding="utf-8").strip()
        return int(value) if value else 1
    except Exception:
        return 1


def save_next_id(v: int) -> None:
    """Persist the next `newsid` cursor to disk."""
    NEXT_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    NEXT_ID_FILE.write_text(str(int(v)), encoding="utf-8")


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

        time.sleep(random.uniform(parser.min_delay, parser.max_delay))

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

    kp = KndcParser(output_dir=output_dir)

    # hourly bulletin parser (runs each scheduler cycle; writes local snapshots)
    bp = KndcBulletinHourlyParser(output_dir=output_dir, limit=25)

    while True:
        try:
            seen = load_seen()
            next_id = load_next_id()

            logger.info("Starting probe. next_id=%s seen=%d", next_id, len(seen))

            # Run hourly bulletin parser first to capture any new bulletins
            try:
                new_bulletins = bp.run_once()
                if new_bulletins:
                    logger.info(
                        "Hourly bulletin parser saved %d new bulletin(s)", new_bulletins
                    )
            except Exception:
                logger.exception("Bulletin parser error")

            discovered, next_id, advanced = probe_new_ids(
                kp, next_id, max_per_run=max_per_run
            )
            save_next_id(next_id)

            # Deduplicate and keep only those not seen before
            new_records = [(nid, rec) for nid, rec in discovered if nid not in seen]

            if new_records:
                new_ids = [nid for nid, _ in new_records]
                logger.info("Discovered %d new id(s): %s", len(new_ids), new_ids)
                records = []
                for _nid, rec in new_records:
                    records.append(rec)
                    p = kp.save(rec)
                    logger.info("Saved %s", p)
                seen.update(new_ids)
                save_seen(seen)
                logger.info("Saved %d record(s)", len(records))
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

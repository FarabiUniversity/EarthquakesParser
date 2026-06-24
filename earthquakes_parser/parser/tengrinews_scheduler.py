"""Hourly scheduler for TengriNews parser.

Usage::

    python -m earthquakes_parser.parser.tengrinews_scheduler

The process runs forever, triggering a full parse cycle every hour.
Stop it with Ctrl-C or SIGTERM.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from earthquakes_parser.parser.tengrinews_parser import TengriNewsParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = 3600  # 1 hour


def _handle_signal(signum: int, _frame: object) -> None:
    logger.info("Received signal %d – shutting down.", signum)
    sys.exit(0)


def run_scheduler(
    output_dir: str = "data/tengrinews",
    interval: int = _INTERVAL_SECONDS,
) -> None:
    """Start the hourly parsing loop.

    Args:
        output_dir: Directory where parsed JSON files are saved.
        interval: Seconds between parse cycles (default 3600).
    """
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    parser = TengriNewsParser(output_dir=output_dir)
    logger.info(
        "TengriNews scheduler started. Interval: %ds. Output: %s",
        interval,
        output_dir,
    )

    while True:
        logger.info("=== Starting parse cycle ===")
        try:
            articles = parser.run()
            logger.info("=== Cycle complete – %d new article(s) ===", len(articles))
        except Exception as exc:
            logger.exception("Unhandled error during parse cycle: %s", exc)

        logger.info("Next cycle in %d seconds.", interval)
        time.sleep(interval)


if __name__ == "__main__":
    run_scheduler()

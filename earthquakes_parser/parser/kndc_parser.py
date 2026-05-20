"""KNDC news parser.

Fetches individual KNDC news items by numeric `newsid` and stores snapshots.
"""

from __future__ import annotations

import json
import logging
import random
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Android 14; Mobile; rv:125.0) Gecko/125.0 Firefox/125.0",
]


@dataclass
class KndcRecord:
    """A normalized KNDC record extracted from the site."""

    source: str
    source_url: str
    title: str
    published_at: Optional[str]
    main_text: str
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record into a JSON-safe dictionary."""
        return {
            "source": self.source,
            "source_url": self.source_url,
            "title": self.title,
            "published_at": self.published_at,
            "main_text": self.main_text,
            "raw": self.raw,
        }


class KndcParser:
    """Simple KNDC parser for fetching single news items by `newsid`.

    Designed to be used as a lightweight one-shot parser called from a
    scheduler script or cron job. It intentionally focuses on deterministic
    fetching and cleaning rather than full site crawling.
    """

    AJAX_URL = "https://kndc.kz/modules/mod_kndcnews/ajax/getNews.php"
    PAGE_URL = "https://kndc.kz/index.php?news={newsid}"

    def __init__(
        self,
        output_dir: str = "data/kndc",
        min_delay: float = 0.5,
        max_delay: float = 1.5,
        max_retries: int = 3,
        timeout: int = 15,
    ) -> None:
        """Initialize the parser and its HTTP session."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._min_delay = float(min_delay)
        self._max_delay = float(max_delay)
        self._max_retries = int(max_retries)
        self._timeout = int(timeout)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "ru-RU,ru;q=0.9,kk;q=0.8,en-US;q=0.7,en;q=0.5",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Referer": "https://kndc.kz/",
                "X-Requested-With": "XMLHttpRequest",
            }
        )

    @property
    def min_delay(self) -> float:
        """Minimum delay between requests in seconds."""
        return self._min_delay

    @property
    def max_delay(self) -> float:
        """Maximum delay between requests in seconds."""
        return self._max_delay

    def _build_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "X-Requested-With": "XMLHttpRequest",
        }

    def _get(self, params: Dict[str, Any]) -> Optional[requests.Response]:
        headers = self._build_headers()

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(
                    self.AJAX_URL, params=params, headers=headers, timeout=self._timeout
                )
                if resp.status_code == 200:
                    return resp

                if resp.status_code == 429:
                    wait = 5 * attempt
                    logger.warning(
                        "Rate limited on %s. Sleeping %ds", self.AJAX_URL, wait
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2**attempt
                    logger.warning(
                        "Server error %d fetching %s. Retrying in %ds",
                        resp.status_code,
                        self.AJAX_URL,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                logger.warning(
                    "Unexpected status %d fetching %s", resp.status_code, self.AJAX_URL
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Request error (attempt %d/%d): %s", attempt, self._max_retries, exc
                )
                time.sleep(2**attempt)

        logger.error("Failed to fetch after %d attempts: %s", self._max_retries, params)
        return None

    def _clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html or "", "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return unicodedata.normalize("NFKD", text)

    def fetch_by_newsid(self, newsid: int) -> Optional[KndcRecord]:
        """Fetch and parse a single KNDC news item by `newsid`."""
        resp = self._get({"newsid": int(newsid)})
        if not resp:
            return None

        try:
            raw = resp.json()
        except ValueError:
            logger.exception("Invalid JSON for newsid=%s", newsid)
            return None

        params = raw.get("params")
        try:
            params_dict = (
                json.loads(params) if isinstance(params, str) else params or {}
            )
        except Exception:
            params_dict = {}

        title = params_dict.get("title", "")
        published_at = raw.get("lddate")
        details = raw.get("details", "")

        if not any([title, published_at, details]):
            logger.info("Empty KNDC payload for newsid=%s; skipping", newsid)
            return None

        main_text = self._clean_html(details)

        rec = KndcRecord(
            source="kndc",
            source_url=self.PAGE_URL.format(newsid=newsid),
            title=title,
            published_at=published_at,
            main_text=main_text,
            raw=raw,
        )

        return rec

    def save(self, record: KndcRecord, filename: Optional[str] = None) -> Path:
        """Persist a record snapshot to `output_dir` and return the file path."""
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        nid = "unknown"
        try:
            if "news=" in record.source_url:
                nid = record.source_url.split("news=")[-1]
        except Exception:
            nid = "unknown"

        name = filename or f"kndc_{nid}_{ts}.json"
        out = self.output_dir / name
        with out.open("w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, ensure_ascii=False, indent=2)
        return out

    def run(self, newsids: List[int], save: bool = True) -> List[KndcRecord]:
        """Fetch a batch of `newsids` and optionally save them to disk."""
        out: List[KndcRecord] = []
        for nid in newsids:
            rec = self.fetch_by_newsid(nid)
            if rec:
                out.append(rec)
                if save:
                    p = self.save(rec)
                    logger.info("Saved %s", p)
            else:
                logger.debug("Skipping empty KNDC newsid=%s", nid)
            time.sleep(random.uniform(self._min_delay, self._max_delay))
        return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="KNDC parser")
    parser.add_argument(
        "--newsids", default="3191", help="Comma-separated news ids to fetch"
    )
    parser.add_argument("--output", default="data/kndc", help="Output directory")
    args = parser.parse_args()

    ids = [int(x.strip()) for x in args.newsids.split(",") if x.strip()]
    kp = KndcParser(output_dir=args.output)
    recs = kp.run(ids)
    print(f"Fetched {len(recs)} record(s)")

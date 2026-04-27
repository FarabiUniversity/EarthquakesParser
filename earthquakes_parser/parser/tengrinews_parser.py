"""TengriNews parser – scrapes news articles from https://tengrinews.kz/news/."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_LISTING_URL = "https://tengrinews.kz/news/"
_BASE_URL = "https://tengrinews.kz"

# Realistic browser User-Agent pool – rotated per request to reduce fingerprinting.
_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 14; Mobile; rv:125.0) Gecko/125.0 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Article links on the listing page live inside span.content_main_item_title > a.
# The listing uses /category/slug-NNNNN/ URLs (not /news/), so we match by slug pattern.
_LINK_SELECTOR = "span.content_main_item_title a"

# Regex that matches a valid article path (ends with a 5+ digit ID).
_ARTICLE_PATH_RE = re.compile(r"^/[a-z_-]+/[a-z0-9-]+-\d{5,}/?$")

# CSS selectors for article-page fields (tried in order; first match wins).
_TITLE_SELECTORS = [
    "h1.head-single",
    "h1",
]

# The breadcrumb date span contains the full explicit date, e.g. "27 апреля 2026 17:42".
_DATE_SELECTORS = [
    "span.breadcrumbs__item--date",
    "time[datetime]",
]

_BODY_SELECTORS = [
    "div.content_main_text",          # has itemprop="articleBody"
    "div[itemprop='articleBody']",
]

# Russian month names → month number.
_RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

# Noise elements inside the article body that should be stripped before text extraction.
_BODY_NOISE_SELECTORS = "script, style, .tn-inpage, .tn-discussed-now-block, .social-share, .tags"


def _parse_ru_datetime(text: str) -> Optional[str]:
    """Parse a Russian-language date string into ISO-8601 format.

    Handles two formats produced by tengrinews.kz:
    - Explicit: "27 апреля 2026 17:42"
    - Relative: "Сегодня 17:42" / "Вчера 17:42"

    Returns an ISO string like "2026-04-27T17:42:00", or None on failure.
    """
    text = text.strip()

    # Explicit: "27 апреля 2026 17:42"
    m = re.match(
        r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})\s+(\d{1,2}):(\d{2})",
        text,
        re.IGNORECASE,
    )
    if m:
        day, month_ru, year, hour, minute = m.groups()
        month = _RU_MONTHS.get(month_ru.lower())
        if month:
            try:
                dt = datetime(int(year), month, int(day), int(hour), int(minute))
                return dt.isoformat(timespec="seconds")
            except ValueError:
                pass

    # Relative: "Сегодня 17:42" or "Вчера 17:42"
    m = re.match(r"(Сегодня|Вчера)\s+(\d{1,2}):(\d{2})", text, re.IGNORECASE)
    if m:
        word, hour, minute = m.groups()
        from datetime import timedelta

        base = datetime.now().date()
        if word.lower() == "вчера":
            base = base - timedelta(days=1)
        try:
            dt = datetime(base.year, base.month, base.day, int(hour), int(minute))
            return dt.isoformat(timespec="seconds")
        except ValueError:
            pass

    return None


@dataclass
class NewsArticle:
    """Single parsed news article."""

    url: str
    title: str
    published_at: str
    main_text: str

    def to_dict(self) -> dict:
        """Return plain dict representation."""
        return asdict(self)


class TengriNewsParser:
    """Fetches and parses news articles from tengrinews.kz.

    Anti-blocking measures:
    - Random User-Agent per request.
    - Random delay (1.5–4 s) between requests.
    - Retry with exponential back-off on 429 / 5xx.
    - Realistic browser headers (Accept, Referer, etc.).
    - Persistent requests.Session for TCP keep-alive.
    """

    def __init__(
        self,
        output_dir: str = "data/tengrinews",
        min_delay: float = 1.5,
        max_delay: float = 4.0,
        max_retries: int = 3,
        timeout: int = 20,
    ) -> None:
        """Initialise parser.

        Args:
            output_dir: Directory where JSON result files are written.
            min_delay: Minimum seconds to sleep between HTTP requests.
            max_delay: Maximum seconds to sleep between HTTP requests.
            max_retries: Maximum retry attempts for a single request.
            timeout: HTTP request timeout in seconds.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._seen_file = self.output_dir / "seen_urls.txt"
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._max_retries = max_retries
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[NewsArticle]:
        """Fetch all new articles and persist them.

        Returns:
            List of newly parsed articles (empty when nothing is new).
        """
        seen = self._load_seen_urls()
        links = self._fetch_article_links()
        new_links = [u for u in links if u not in seen]

        if not new_links:
            logger.info("No new articles found.")
            return []

        logger.info("Found %d new article(s) to parse.", len(new_links))

        articles: List[NewsArticle] = []
        random.shuffle(new_links)  # vary crawl order

        for url in new_links:
            article = self._parse_article(url)
            if article:
                articles.append(article)
                seen.add(url)
            self._sleep()

        if articles:
            self._save_articles(articles)
            self._persist_seen_urls(seen)

        return articles

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, referer: Optional[str] = None) -> Optional[requests.Response]:
        """GET *url* with retry logic and anti-blocking headers.

        Args:
            url: Target URL.
            referer: Value for the Referer header (optional).

        Returns:
            Response object or None if all retries failed.
        """
        headers = self._build_headers(referer or _LISTING_URL)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(
                    url, headers=headers, timeout=self._timeout
                )

                if response.status_code == 200:
                    return response

                if response.status_code == 429:
                    wait = 30 * attempt
                    logger.warning("Rate-limited (429). Sleeping %ds.", wait)
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = 5 * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error %d. Retrying in %ds (attempt %d/%d).",
                        response.status_code,
                        wait,
                        attempt,
                        self._max_retries,
                    )
                    time.sleep(wait)
                    continue

                logger.warning("Unexpected status %d for %s", response.status_code, url)
                return None

            except requests.exceptions.Timeout:
                logger.warning("Timeout on %s (attempt %d/%d).", url, attempt, self._max_retries)
                time.sleep(3 * attempt)

            except requests.exceptions.RequestException as exc:
                logger.error("Request error for %s: %s", url, exc)
                return None

        logger.error("All %d retries exhausted for %s.", self._max_retries, url)
        return None

    @staticmethod
    def _build_headers(referer: str) -> dict:
        """Return a dict of realistic browser headers with a random User-Agent."""
        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,kk;q=0.8,en-US;q=0.7,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": referer,
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

    def _sleep(self) -> None:
        """Sleep a random interval between requests."""
        time.sleep(random.uniform(self._min_delay, self._max_delay))

    # ------------------------------------------------------------------
    # Scraping logic
    # ------------------------------------------------------------------

    def _fetch_article_links(self) -> List[str]:
        """Scrape all article URLs from the listing page.

        Returns:
            Deduplicated list of absolute article URLs.
        """
        response = self._get(_LISTING_URL)
        if not response:
            logger.error("Failed to fetch listing page.")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        urls: List[str] = []

        for tag in soup.select(_LINK_SELECTOR):
            href = tag.get("href", "")
            if href and _ARTICLE_PATH_RE.match(href):
                full_url = urljoin(_BASE_URL, href)
                if full_url not in urls:
                    urls.append(full_url)

        logger.info("Collected %d article links from listing.", len(urls))
        return urls

    def _parse_article(self, url: str) -> Optional[NewsArticle]:
        """Download and parse a single article page.

        Args:
            url: Article URL.

        Returns:
            Parsed NewsArticle or None on failure.
        """
        response = self._get(url, referer=_LISTING_URL)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = self._extract_text(soup, _TITLE_SELECTORS)
        if not title:
            logger.warning("Could not extract title from %s", url)
            return None

        published_at = self._extract_date(soup)
        main_text = self._extract_body(soup)

        return NewsArticle(
            url=url,
            title=title.strip(),
            published_at=published_at,
            main_text=main_text.strip(),
        )

    @staticmethod
    def _extract_text(soup: BeautifulSoup, selectors: List[str]) -> str:
        """Try selectors in order; return first non-empty text found."""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    return text
        return ""

    @staticmethod
    def _extract_date(soup: BeautifulSoup) -> str:
        """Extract publication date/time as ISO-8601 string.

        The breadcrumb span contains the explicit full date like
        "27 апреля 2026 17:42" which is parsed into ISO format.
        A <time datetime="..."> attribute is used directly when present.
        """
        for selector in _DATE_SELECTORS:
            element = soup.select_one(selector)
            if not element:
                continue
            dt_attr = element.get("datetime")
            if dt_attr:
                return str(dt_attr)
            text = element.get_text(separator=" ", strip=True)
            if text:
                iso = _parse_ru_datetime(text)
                return iso if iso else text
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        """Extract main article body text, stripping ads and noise."""
        for selector in _BODY_SELECTORS:
            element = soup.select_one(selector)
            if element:
                for noise in element.select(_BODY_NOISE_SELECTORS):
                    noise.decompose()
                text = element.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    return text
        return ""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_seen_urls(self) -> set:
        """Load the set of already-processed article URLs from disk."""
        if self._seen_file.exists():
            return set(self._seen_file.read_text(encoding="utf-8").splitlines())
        return set()

    def _persist_seen_urls(self, urls: set) -> None:
        """Write the updated seen-URLs set to disk."""
        self._seen_file.write_text("\n".join(sorted(urls)), encoding="utf-8")

    def _save_articles(self, articles: List[NewsArticle]) -> None:
        """Append new articles to a dated JSON file.

        Args:
            articles: Articles to persist.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_file = self.output_dir / f"tengrinews_{timestamp}.json"
        payload = [a.to_dict() for a in articles]

        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        logger.info("Saved %d article(s) to %s.", len(articles), out_file)

"""HTML downloader utility supporting requests (bs4) and Selenium-based fetching."""

from typing import Literal
from urllib.parse import urlparse

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class HTMLDownloader:
    """Download HTML content using either requests (bs4) or Selenium."""

    def __init__(self, fetch_with: Literal["bs4", "selenium"] = "selenium"):
        """Initialize the downloader with chosen fetch method (bs4 or selenium)."""
        self.fetch_with = fetch_with

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the provided URL is valid (http/https with a netloc)."""
        parsed = urlparse(url)
        return all([parsed.scheme in ("http", "https"), parsed.netloc])

    def fetch_html(self, url: str) -> str:
        """Fetch HTML from a given URL using the configured method."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        if self.fetch_with == "selenium":
            return self._fetch_with_selenium(url)
        elif self.fetch_with == "bs4":
            return self._fetch_with_bs4(url)
        else:
            raise ValueError(f"Unsupported fetch method: {self.fetch_with}")

    @staticmethod
    def _fetch_with_bs4(url: str, timeout: int = 10) -> str:
        """Fetch HTML using requests (bs4)."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise RuntimeError(f"[BS4] Failed to fetch {url}: {e}") from e

    @staticmethod
    def _fetch_with_selenium(url: str, timeout: int = 10) -> str:
        """Fetch HTML using Selenium with a headless Chrome driver."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            html: str = driver.page_source
            return html
        except Exception as e:
            raise RuntimeError(f"[Selenium] Failed to fetch {url}: {e}") from e
        finally:
            if driver:
                driver.quit()

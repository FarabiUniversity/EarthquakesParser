"""Data extraction from HTML using schemas."""

from typing import List, Optional

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from earthquakes_parser.parser.models import ExtractionResult, PageSchema


class DataExtractor:
    """Extracts data from HTML using page schemas."""

    def extract(self, html: str, schema: PageSchema) -> ExtractionResult:
        """Extract data from HTML using schema.

        Args:
            html: HTML content.
            schema: PageSchema with selectors.

        Returns:
            ExtractionResult with extracted data.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract main text
            main_texts = self._extract_main_text(soup, schema.main_text_selectors)

            # Extract date
            date = self._extract_date(soup, schema.date_selector)

            # Check if extraction was successful
            success = bool(main_texts) or date is not None

            return ExtractionResult(
                main_text=main_texts,
                date=date,
                success=success,
            )

        except Exception as e:
            print(f"❌ Error extracting data: {e}")
            return ExtractionResult(
                main_text=[],
                date=None,
                success=False,
                error=str(e),
            )

    def _extract_main_text(
            self, soup: BeautifulSoup, selectors: List[str]
    ) -> List[str]:
        """Extract main text using CSS selectors.

        Args:
            soup: BeautifulSoup object.
            selectors: List of CSS selectors.

        Returns:
            List of extracted text strings.
        """
        texts = []

        for selector in selectors:
            try:
                elements = soup.select(selector)
                for el in elements:
                    text = el.get_text(strip=True)
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"⚠️ Error with selector '{selector}': {e}")

        return texts

    def _extract_date(
            self, soup: BeautifulSoup, selector: Optional[str]
    ) -> Optional[str]:
        """Extract and parse date using CSS selector.

        Args:
            soup: BeautifulSoup object.
            selector: CSS selector for date element.

        Returns:
            ISO formatted date string or None.
        """
        if not selector:
            return None

        try:
            elements = soup.select(selector)
            if not elements:
                return None

            date_text = elements[0].get_text(strip=True)
            if not date_text:
                return None

            # Parse date
            parsed_date = date_parser.parse(date_text, fuzzy=True).date()
            return parsed_date.isoformat()  # YYYY-MM-DD format

        except Exception as e:
            print(f"⚠️ Failed to parse date: {e}")
            return None

    def validate_extraction(self, result: ExtractionResult) -> bool:
        """Validate if extraction result is valid.

        Args:
            result: ExtractionResult to validate.

        Returns:
            True if both main_text and date are empty/None (need re-extraction).
        """
        return not result.main_text and result.date is None
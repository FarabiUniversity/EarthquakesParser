"""Data models for parser module."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class PageSchema:
    """Schema for extracting data from a website."""

    domain: str
    main_text_selectors: List[str]
    date_selector: Optional[str]
    is_valid: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for database insertion."""
        return {
            "domain": self.domain,
            "main_text_selectors": self.main_text_selectors,
            "date_selector": self.date_selector,
            "is_valid": self.is_valid,
        }


@dataclass
class ParsedContent:
    """Extracted content from a webpage."""

    search_result_id: str
    main_text: List[str]
    date: Optional[str]
    page_schema_id: str
    parsed_at: Optional[datetime] = None
    id: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for database insertion."""
        return {
            "search_result_id": self.search_result_id,
            "main_text": self.main_text,
            "date": self.date,
            "page_schema_id": self.page_schema_id,
        }


@dataclass
class ExtractionResult:
    """Result of data extraction from HTML."""

    main_text: List[str]
    date: Optional[str]
    success: bool
    error: Optional[str] = None
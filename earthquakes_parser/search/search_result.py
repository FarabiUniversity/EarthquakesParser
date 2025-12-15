"""Data model for representing search results in the earthquakes_parser project."""

from typing import Optional


class SearchResult:
    """Represents a single search result."""

    def __init__(self, query: str, link: str, title: Optional[str] = None):
        """Initialize a search result.

        Args:
            query: The search query used.
            link: URL of the search result.
            title: Title of the search result page.
        """
        self.query = query
        self.link = link
        self.title = title

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {"query": self.query, "link": self.link, "title": self.title}

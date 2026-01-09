"""Tests for the searcher modules."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from earthquakes_parser import SupabaseDB
from earthquakes_parser.search import DDGSearcher, SearchManager
from earthquakes_parser.search.base_searcher import BaseSearcher
from earthquakes_parser.search.search_result import SearchResult


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            query="earthquake", link="https://example.com", title="Test"
        )
        assert result.query == "earthquake"
        assert result.link == "https://example.com"
        assert result.title == "Test"

    def test_to_dict(self):
        """Test converting SearchResult to dictionary."""
        result = SearchResult(
            query="earthquake", link="https://example.com", title="Test"
        )
        data = result.to_dict()
        assert data["query"] == "earthquake"
        assert data["link"] == "https://example.com"
        assert data["title"] == "Test"


class TestDDGSearcher:
    """Tests for DDGSearcher class."""

    def test_searcher_initialization(self):
        """Test searcher initialization."""
        searcher = DDGSearcher(delay=0.1)
        assert searcher.delay == 0.1
        assert searcher.ddgs is not None

    def test_load_keywords_from_file(self, tmp_path):
        """Test loading keywords from file."""
        keywords_file = tmp_path / "keywords.txt"
        keywords_file.write_text("keyword1\nkeyword2\nkeyword3\n")

        keywords = DDGSearcher.load_keywords_from_file(str(keywords_file))

        assert len(keywords) == 3
        assert "keyword1" in keywords
        assert "keyword2" in keywords


class MockSearcher(BaseSearcher):
    """Simple mock searcher for testing."""

    def __init__(self):
        self.results = []

    def search(self, query, max_results=5, site_filter=None, offset=0):
        """Return preset results."""
        return self.results


class TestSearchManager:
    """Tests for SearchManager class."""

    def test_initialization(self):
        """Test SearchManager initialization."""
        mock_db = MagicMock(spec=SupabaseDB)
        mock_searcher = MockSearcher()
        manager = SearchManager(db=mock_db, searcher=mock_searcher)

        assert manager.db == mock_db
        assert manager.searcher == mock_searcher

    def test_search_and_save_basic(self):
        """Test basic search and save."""
        mock_db = MagicMock(spec=SupabaseDB)
        mock_searcher = MockSearcher()
        manager = SearchManager(db=mock_db, searcher=mock_searcher)

        # Setup
        mock_searcher.results = [
            SearchResult("test", "https://example.com/1", "Title 1")
        ]
        mock_db.exists.return_value = False
        mock_db.insert.return_value = ["id1"]

        # Execute
        stats = manager.search_and_save(["test"], max_results=1)

        # Verify
        assert stats["searched"] == 1
        assert stats["new"] == 1
        assert stats["skipped"] == 0

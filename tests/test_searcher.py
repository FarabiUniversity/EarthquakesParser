"""Tests for the SearchManager module."""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from earthquakes_parser import SupabaseDB, SupabaseFileStorage
from earthquakes_parser.search import SearchManager
from earthquakes_parser.search.base_searcher import BaseSearcher
from earthquakes_parser.search.search_result import SearchResult


class MockSearcher(BaseSearcher):
    """Mock implementation of BaseSearcher for testing."""

    def __init__(self, mock_results=None):
        self.mock_results = mock_results or []
        self.search_calls = []

    def search(self, query, max_results=5, site_filter=None, offset=0):
        """Mock search implementation."""
        self.search_calls.append(
            {
                "query": query,
                "max_results": max_results,
                "site_filter": site_filter,
                "offset": offset,
            }
        )
        return self.mock_results


class TestSearchManager:
    """Tests for SearchManager class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock SupabaseDB instance."""
        db = MagicMock(spec=SupabaseDB)
        return db

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock searcher instance."""
        return MockSearcher()

    @pytest.fixture
    def search_manager(self, mock_db, mock_searcher):
        """Create a SearchManager instance with mocks."""
        return SearchManager(db=mock_db, searcher=mock_searcher)

    def test_initialization(self, mock_db, mock_searcher):
        """Test SearchManager initialization."""
        manager = SearchManager(db=mock_db, searcher=mock_searcher)
        assert manager.db == mock_db
        assert manager.searcher == mock_searcher

    def test_search_and_save_basic(self, search_manager, mock_db, mock_searcher):
        """Test basic search and save functionality."""
        # Setup mock results
        mock_searcher.mock_results = [
            SearchResult(
                query="землетрясение",
                link="https://example.com/1",
                title="Article 1",
            ),
            SearchResult(
                query="землетрясение",
                link="https://example.com/2",
                title="Article 2",
            ),
        ]

        # Mock database methods
        mock_db.exists.return_value = False
        mock_db.insert.return_value = ["id1", "id2"]

        # Execute search
        stats = search_manager.search_and_save(
            keywords=["землетрясение"],
            max_results=2,
        )

        # Verify results
        assert stats["searched"] == 1
        assert stats["found"] == 2
        assert stats["new"] == 2
        assert stats["skipped"] == 0

        # Verify database calls
        assert mock_db.insert.call_count == 1
        inserted_data = mock_db.insert.call_args[0][1]
        assert len(inserted_data) == 2
        assert inserted_data[0]["query"] == "землетрясение"
        assert inserted_data[0]["link"] == "https://example.com/1"
        assert inserted_data[0]["status"] == "pending"

    def test_search_and_save_with_duplicates(
        self, search_manager, mock_db, mock_searcher
    ):
        """Test search and save skipping duplicate URLs."""
        mock_searcher.mock_results = [
            SearchResult(
                query="earthquake",
                link="https://example.com/1",
                title="New",
            ),
            SearchResult(
                query="earthquake",
                link="https://example.com/2",
                title="Duplicate",
            ),
            SearchResult(
                query="earthquake",
                link="https://example.com/3",
                title="Another",
            ),
        ]

        # Mock: second URL already exists
        mock_db.exists.side_effect = [False, True, False]
        mock_db.insert.return_value = ["id1", "id3"]

        stats = search_manager.search_and_save(
            keywords=["earthquake"],
            max_results=3,
            skip_existing=True,
        )

        assert stats["found"] == 3
        assert stats["new"] == 2
        assert stats["skipped"] == 1

        # Verify only non-duplicate URLs were inserted
        inserted_data = mock_db.insert.call_args[0][1]
        assert len(inserted_data) == 2
        assert inserted_data[0]["link"] == "https://example.com/1"
        assert inserted_data[1]["link"] == "https://example.com/3"

    def test_search_and_save_with_site_filter(
        self, search_manager, mock_db, mock_searcher
    ):
        """Test search with site filter."""
        mock_searcher.mock_results = [
            SearchResult(
                query="earthquake",
                link="https://instagram.com/post1",
                title="IG Post",
            ),
        ]
        mock_db.exists.return_value = False
        mock_db.insert.return_value = ["id1"]

        stats = search_manager.search_and_save(
            keywords=["earthquake"],
            max_results=1,
            site_filter="instagram.com",
        )

        assert stats["new"] == 1
        inserted_data = mock_db.insert.call_args[0][1]
        assert inserted_data[0]["site_filter"] == "instagram.com"

    def test_search_and_save_pagination(self, search_manager, mock_db, mock_searcher):
        """Test that searcher continues with offset when needed."""
        # First batch: 2 results (need 5 total)
        first_batch = [
            SearchResult(
                query="test",
                link=f"https://example.com/{i}",
                title=f"Article {i}",
            )
            for i in range(1, 3)
        ]
        # Second batch: 3 more results
        second_batch = [
            SearchResult(
                query="test",
                link=f"https://example.com/{i}",
                title=f"Article {i}",
            )
            for i in range(3, 6)
        ]

        # Setup mock to return different results on consecutive calls
        mock_searcher.mock_results = first_batch

        def search_side_effect(query, max_results, site_filter, offset):
            if offset == 1:
                return first_batch
            elif offset == 11:
                return second_batch
            return []

        mock_searcher.search = Mock(side_effect=search_side_effect)
        mock_db.exists.return_value = False
        mock_db.insert.return_value = [f"id{i}" for i in range(100)]

        stats = search_manager.search_and_save(
            keywords=["test"],
            max_results=5,
        )

        # Verify pagination occurred
        assert mock_searcher.search.call_count == 2
        assert stats["new"] == 5

    def test_search_and_save_no_results(self, search_manager, mock_db, mock_searcher):
        """Test search when no results are found."""
        mock_searcher.mock_results = []

        stats = search_manager.search_and_save(
            keywords=["nonexistent"],
            max_results=5,
        )

        assert stats["searched"] == 1
        assert stats["found"] == 0
        assert stats["new"] == 0
        assert mock_db.insert.call_count == 0

    def test_search_and_save_multiple_keywords(
        self, search_manager, mock_db, mock_searcher
    ):
        """Test search with multiple keywords."""
        mock_searcher.mock_results = [
            SearchResult(
                query="keyword",
                link="https://example.com/1",
                title="Article 1",
            ),
        ]
        mock_db.exists.return_value = False
        mock_db.insert.return_value = ["id1"]

        stats = search_manager.search_and_save(
            keywords=["keyword1", "keyword2", "keyword3"],
            max_results=1,
        )

        assert stats["searched"] == 3
        assert mock_db.insert.call_count == 3

    def test_get_urls_with_pending_status(self, search_manager, mock_db):
        """Test getting URLs with pending status."""
        mock_df = pd.DataFrame(
            [
                {
                    "id": "1",
                    "query": "test",
                    "link": "https://example.com/1",
                    "title": "Title 1",
                    "status": "pending",
                },
                {
                    "id": "2",
                    "query": "test",
                    "link": "https://example.com/2",
                    "title": "Title 2",
                    "status": "pending",
                },
            ]
        )
        mock_db.select.return_value = mock_df

        urls = search_manager.get_urls(status="pending", limit=100)

        assert len(urls) == 2
        assert urls[0]["id"] == "1"
        assert urls[0]["status"] == "pending"
        mock_db.select.assert_called_once_with(
            "search_results", filters={"status": "pending"}, limit=100
        )

    def test_get_urls_empty_result(self, search_manager, mock_db):
        """Test getting URLs when database returns empty dataframe."""
        mock_db.select.return_value = pd.DataFrame()

        urls = search_manager.get_urls(status="pending")

        assert urls == []

    def test_mark_as_success(self, search_manager, mock_db):
        """Test marking a search result with new status."""
        mock_db.update.return_value = {"id": "123", "status": "downloaded"}

        result = search_manager.mark_as("123", "downloaded")

        assert result is True
        mock_db.update.assert_called_once_with(
            "search_results",
            "123",
            {"status": "downloaded"},
        )

    def test_mark_as_failure(self, search_manager, mock_db):
        """Test marking when update fails."""
        mock_db.update.return_value = None

        result = search_manager.mark_as("123", "failed")

        assert result is False

    @patch("earthquakes_parser.search.search_manager.HTMLDownloader")
    def test_download_html_success(
        self, mock_downloader_class, search_manager, mock_db
    ):
        """Test successful HTML download."""
        # Setup mocks
        mock_storage = MagicMock(spec=SupabaseFileStorage)
        mock_downloader = MagicMock()
        mock_downloader_class.return_value = mock_downloader

        mock_db.select.return_value = pd.DataFrame(
            [
                {
                    "id": "1",
                    "link": "https://example.com/1",
                    "status": "pending",
                },
            ]
        )

        mock_downloader.fetch_html.return_value = "<html>Content</html>"
        mock_storage.upload.return_value = "storage/1.html"
        mock_db.update.return_value = {"id": "1"}

        stats = search_manager.download_html(
            storage=mock_storage,
            fetch_with="bs4",
            limit=50,
        )

        assert stats["downloaded"] == 1
        assert stats["failed"] == 0

        # Verify download was called
        mock_downloader.fetch_html.assert_called_once_with("https://example.com/1")

        # Verify storage upload
        mock_storage.upload.assert_called_once_with(
            "1.html", "<html>Content</html>", content_type="text/html"
        )

    @patch("earthquakes_parser.search.search_manager.HTMLDownloader")
    def test_download_html_fetch_failure(
        self, mock_downloader_class, search_manager, mock_db
    ):
        """Test HTML download when fetch fails."""
        mock_storage = MagicMock(spec=SupabaseFileStorage)
        mock_downloader = MagicMock()
        mock_downloader_class.return_value = mock_downloader

        mock_db.select.return_value = pd.DataFrame(
            [
                {
                    "id": "1",
                    "link": "https://example.com/1",
                    "status": "pending",
                },
            ]
        )

        # Mock empty HTML response (failure)
        mock_downloader.fetch_html.return_value = ""
        mock_db.update.return_value = {"id": "1"}

        stats = search_manager.download_html(
            storage=mock_storage,
            fetch_with="bs4",
            limit=50,
        )

        assert stats["downloaded"] == 0
        assert stats["failed"] == 1

        # Verify status was marked as failed
        assert any(
            call[0][2] == {"status": "failed"} for call in mock_db.update.call_args_list
        )

    @patch("earthquakes_parser.search.search_manager.HTMLDownloader")
    def test_download_html_upload_failure(
        self, mock_downloader_class, search_manager, mock_db
    ):
        """Test HTML download when storage upload fails."""
        mock_storage = MagicMock(spec=SupabaseFileStorage)
        mock_downloader = MagicMock()
        mock_downloader_class.return_value = mock_downloader

        mock_db.select.return_value = pd.DataFrame(
            [
                {
                    "id": "1",
                    "link": "https://example.com/1",
                    "status": "pending",
                },
            ]
        )

        mock_downloader.fetch_html.return_value = "<html>Content</html>"
        mock_storage.upload.return_value = None  # Upload failure
        mock_db.update.return_value = {"id": "1"}

        stats = search_manager.download_html(
            storage=mock_storage,
            fetch_with="selenium",
            limit=50,
        )

        assert stats["downloaded"] == 0
        assert stats["failed"] == 1

    def test_get_statistics(self, search_manager, mock_db):
        """Test getting search statistics."""
        mock_df = pd.DataFrame(
            [
                {"status": "pending"},
                {"status": "pending"},
                {"status": "downloaded"},
                {"status": "parsed"},
                {"status": "failed"},
            ]
        )
        mock_db.select.return_value = mock_df

        stats = search_manager.get_statistics()

        assert stats["total"] == 5
        assert stats["pending"] == 2
        assert stats["downloaded"] == 1
        assert stats["parsed"] == 1
        assert stats["analyzed"] == 0
        assert stats["failed"] == 1

    def test_get_statistics_empty(self, search_manager, mock_db):
        """Test statistics with empty database."""
        mock_db.select.return_value = pd.DataFrame(columns=["status"])

        stats = search_manager.get_statistics()

        assert stats["total"] == 0
        assert all(
            stats[key] == 0
            for key in [
                "pending",
                "downloaded",
                "parsed",
                "analyzed",
                "failed",
            ]
        )

    @patch(
        "earthquakes_parser.search.base_searcher."
        "BaseSearcher.load_keywords_from_file"
    )
    def test_search_with_keywords_file(
        self, mock_load, search_manager, mock_db, mock_searcher
    ):
        """Test search using keywords from file."""
        mock_load.return_value = ["keyword1", "keyword2"]
        mock_searcher.mock_results = [
            SearchResult(
                query="keyword",
                link="https://example.com/1",
                title="Article",
            ),
        ]
        mock_db.exists.return_value = False
        mock_db.insert.return_value = ["id1"]

        stats = search_manager.search_with_keywords_file(
            keywords_file="/path/to/keywords.txt",
            max_results=1,
        )

        mock_load.assert_called_once_with("/path/to/keywords.txt")
        assert stats["searched"] == 2

    def test_search_and_save_skip_existing_disabled(
        self, search_manager, mock_db, mock_searcher
    ):
        """Test search without skipping existing URLs."""
        mock_searcher.mock_results = [
            SearchResult(query="test", link="https://example.com/1", title="Article"),
        ]
        mock_db.insert.return_value = ["id1"]

        stats = search_manager.search_and_save(
            keywords=["test"],
            max_results=1,
            skip_existing=False,
        )

        # exists() should not be called when skip_existing=False
        mock_db.exists.assert_not_called()
        assert stats["skipped"] == 0
        assert stats["new"] == 1


class TestSearchManagerIntegration:
    """Integration tests for SearchManager with realistic scenarios."""

    @pytest.fixture
    def integration_manager(self):
        """Create manager with realistic mock setup."""
        db = MagicMock(spec=SupabaseDB)
        searcher = MockSearcher()
        return SearchManager(db=db, searcher=searcher), db, searcher

    def test_full_workflow(self, integration_manager):
        """Test complete workflow: search -> download -> statistics."""
        manager, mock_db, mock_searcher = integration_manager

        # Step 1: Search and save
        mock_searcher.mock_results = [
            SearchResult(
                query="earthquake",
                link="https://example.com/1",
                title="Article 1",
            ),
            SearchResult(
                query="earthquake",
                link="https://example.com/2",
                title="Article 2",
            ),
        ]
        mock_db.exists.return_value = False
        mock_db.insert.return_value = ["id1", "id2"]

        search_stats = manager.search_and_save(["earthquake"], max_results=2)
        assert search_stats["new"] == 2

        # Step 2: Get pending URLs
        mock_db.select.return_value = pd.DataFrame(
            [
                {
                    "id": "id1",
                    "link": "https://example.com/1",
                    "status": "pending",
                },
                {
                    "id": "id2",
                    "link": "https://example.com/2",
                    "status": "pending",
                },
            ]
        )

        pending_urls = manager.get_urls(status="pending")
        assert len(pending_urls) == 2

        # Step 3: Mark as downloaded
        mock_db.update.return_value = {"id": "id1"}
        result = manager.mark_as("id1", "downloaded")
        assert result is True

    def test_large_batch_processing(self, integration_manager):
        """Test processing large batch of results."""
        manager, mock_db, mock_searcher = integration_manager

        # Generate 100 mock results
        large_batch = [
            SearchResult(
                query="earthquake",
                link=f"https://example.com/{i}",
                title=f"Article {i}",
            )
            for i in range(100)
        ]
        mock_searcher.mock_results = large_batch
        mock_db.exists.return_value = False
        mock_db.insert.return_value = [f"id{i}" for i in range(100)]

        stats = manager.search_and_save(["earthquake"], max_results=100)

        assert stats["new"] == 100
        assert stats["found"] == 100

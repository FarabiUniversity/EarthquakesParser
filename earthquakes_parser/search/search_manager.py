"""Business logic for managing earthquake search operations with Supabase storage."""

from typing import List, Optional, Literal

from earthquakes_parser import SupabaseFileStorage
from earthquakes_parser.search.html_downloader import HTMLDownloader
from earthquakes_parser.storage.supabase import SupabaseDB
from earthquakes_parser.search.base_searcher import BaseSearcher
from earthquakes_parser.search.search_result import SearchResult


class SearchManager:
    """Manages earthquake search operations with database persistence.

    This class contains business logic for:
    - Searching for earthquake-related content
    - Saving search results to database with deduplication
    - Managing search result status workflow
    """

    def __init__(self, db: SupabaseDB, searcher: BaseSearcher):
        """Initialize search manager.

        Args:
            db: Supabase database utility for persistence.
            searcher: Instance of a searcher implementing BaseSearcher interface.
        """
        self.db = db
        self.searcher = searcher

    def search_and_save(
            self,
            keywords: List[str],
            max_results: int = 5,
            site_filter: Optional[str] = None,
            skip_existing: bool = True,
    ) -> dict:
        """Search for keywords and save results to database.

        Ensures exactly `max_results` new results are saved per keyword,
        skipping duplicates and continuing search with offset if needed.

        Args:
            keywords: List of search keywords.
            max_results: Number of new results to save per keyword.
            site_filter: Optional site filter (e.g., 'instagram.com').
            skip_existing: Skip URLs that already exist in database.

        Returns:
            Dict with statistics: {
                'searched': int,
                'found': int,
                'new': int,
                'skipped': int
            }
        """
        stats = {"searched": 0, "found": 0, "new": 0, "skipped": 0}

        for keyword in keywords:
            stats["searched"] += 1
            collected = []
            offset = 1

            while len(collected) < max_results:
                batch_size = 10
                results = self.searcher.search(
                    query=keyword,
                    max_results=batch_size,
                    site_filter=site_filter,
                    offset=offset
                )

                if not results:
                    break

                stats["found"] += len(results)
                offset += batch_size

                for result in results:
                    if skip_existing and self.db.exists("search_results", "link", result.link):
                        stats["skipped"] += 1
                        continue

                    collected.append({
                        "query": result.query,
                        "link": result.link,
                        "title": result.title,
                        "site_filter": site_filter,
                        "status": "pending",
                    })

                    if len(collected) >= max_results:
                        break

            if collected:
                inserted_ids = self.db.insert("search_results", collected)
                stats["new"] += len(inserted_ids)

        return stats

    def get_urls(self, status: str = "pending", limit: int = 100) -> List[dict]:
        """Get URLs that need to be downloaded.

        Business logic: Fetch records with status.

        Args:
            status: Search status (e.g. 'pending').
            limit: Maximum number of URLs to return.

        Returns:
            List of dicts with keys: id, query, link, title, status.
        """
        df = self.db.select(
            "search_results", filters={"status": status}, limit=limit
        )

        if df.empty:
            return []

        return list(df.to_dict("records"))

    def mark_as(self, search_result_id: str, status: str) -> bool:
        """Mark a search result as downloaded.

        Business logic: Update status from 'pending' to 'downloaded'.

        Args:
            search_result_id: ID of the search result.
            status: Search status (e.g. 'pending').

        Returns:
            True if successful, False otherwise.
        """
        updated = self.db.update(
            "search_results",
            search_result_id,
            {"status": status},
        )
        return updated is not None

    def download_html(
            self,
            storage: SupabaseFileStorage,
            fetch_with: Literal["bs4", "selenium"] = "bs4",
            limit: int = 50
    ) -> dict:
        """Download HTML for pending URLs and upload to Supabase storage.

        Args:
            storage: SupabaseFileStorage instance.
            fetch_with: HTML fetch method: 'bs4' or 'selenium'.
            limit: Max number of URLs to process.

        Returns:
            Dict with stats: {'downloaded': int, 'failed': int}
        """
        stats = {"downloaded": 0, "failed": 0}
        urls = self.get_urls(status="pending", limit=limit)
        downloader = HTMLDownloader(fetch_with=fetch_with)

        for item in urls:
            url = item["link"]
            html = downloader.fetch_html(url)

            if not html.strip():
                self.mark_as(item["id"], "failed")
                stats["failed"] += 1
                continue

            path = f"{item['id']}.html"
            uploaded_path = storage.upload(path, html, content_type="text/html")
            if uploaded_path:
                self.db.update("search_results", item["id"], {
                    "html_storage_path": uploaded_path
                })
                self.mark_as(item["id"], "downloaded")
                stats["downloaded"] += 1
            else:
                self.mark_as(item["id"], "failed")
                stats["failed"] += 1

        return stats

    def get_statistics(self) -> dict:
        """Get search statistics.

        Business logic: Count records by status.

        Returns:
            Dict with counts by status: {
                'total': int,
                'pending': int,
                'downloaded': int,
                'parsed': int,
                'analyzed': int,
                'failed': int
            }
        """
        df = self.db.select("search_results", limit=None)

        counts = df["status"].value_counts()

        stats = {status: counts.get(status, 0) for status in ["pending", "downloaded", "parsed", "analyzed", "failed"]}
        stats["total"] = int(counts.sum())

        return stats

    def search_with_keywords_file(
        self,
        keywords_file: str,
        max_results: int = 5,
        site_filter: Optional[str] = None,
        skip_existing: bool = True,
    ) -> dict:
        """Search using keywords from file and save results.

        Business logic wrapper: Load keywords from file and search.

        Args:
            keywords_file: Path to keywords file (one per line).
            max_results: Maximum results per keyword.
            site_filter: Optional site filter.
            skip_existing: Skip existing URLs.

        Returns:
            Statistics dict from search_and_save().
        """
        keywords = self.searcher.load_keywords_from_file(keywords_file)
        return self.search_and_save(keywords, max_results, site_filter, skip_existing)

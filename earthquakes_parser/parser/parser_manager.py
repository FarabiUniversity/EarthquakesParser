"""Main parser manager orchestrating the parsing workflow."""

from typing import Optional
from urllib.parse import urlparse

from earthquakes_parser.search import GoogleSearcher, DDGSearcher
from earthquakes_parser.storage.supabase.database import SupabaseDB
from earthquakes_parser.parser.data_extractor import DataExtractor
from earthquakes_parser.parser.models import ParsedContent
from earthquakes_parser.parser.schema_extractor import SchemaExtractor
from earthquakes_parser.parser.schema_manager import SchemaManager
from earthquakes_parser.storage.supabase.file_storage import SupabaseFileStorage
from earthquakes_parser.search.search_manager import SearchManager


class ParserManager:
    """Orchestrates the parsing workflow for search results."""

    def __init__(
        self,
        db: SupabaseDB,
        file_storage: SupabaseFileStorage,
        openai_base_url: str = "http://192.168.8.22:9999/v1",
        openai_api_key: str = "api-key",
    ):
        """Initialize parser manager.

        Args:
            db: Database instance.
            file_storage: File storage instance for HTML files.
            openai_base_url: Base URL for OpenAI-compatible API.
            openai_api_key: API key for OpenAI.
        """
        self.db = db
        self.file_storage = file_storage
        self.schema_manager = SchemaManager(db)
        self.schema_extractor = SchemaExtractor(openai_base_url, openai_api_key)
        self.data_extractor = DataExtractor()
        self.parsed_content_table = "parsed_content"

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL.

        Args:
            url: Full URL.

        Returns:
            Domain name (e.g., 'example.com').
        """
        parsed = urlparse(url)
        return parsed.netloc

    def _save_parsed_content(
        self,
        search_result_id: str,
        main_text: list,
        date: Optional[str],
        page_schema_id: Optional[str],
    ) -> Optional[str]:
        """Save parsed content to database.

        Args:
            search_result_id: ID of search result.
            main_text: Extracted main text.
            date: Extracted date.
            page_schema_id: Schema ID used for extraction.

        Returns:
            Parsed content ID or None if failed.
        """
        content = ParsedContent(
            search_result_id=search_result_id,
            main_text=main_text,
            date=date,
            page_schema_id=page_schema_id,
        )

        try:
            inserted_ids = self.db.insert(
                self.parsed_content_table,
                [content.to_dict()],
                batch_size=1,
            )
            return inserted_ids[0] if inserted_ids else None
        except Exception as e:
            print(f"❌ Error saving parsed content: {e}")
            return None

    def update_parsed_content(
            self,
            parsed_content_id: str,
            main_text: Optional[list] = None,
            date: Optional[str] = None,
            page_schema_id: Optional[str] = None,
    ) -> bool:
        """Update parsed content in database.

        Args:
            parsed_content_id: ID of parsed content to update.
            main_text: Updated main text (optional).
            date: Updated date (optional).
            page_schema_id: Updated schema ID (optional).

        Returns:
            True if update succeeded, False otherwise.
        """
        try:
            self.db.update(
                self.parsed_content_table,
                record_id=parsed_content_id,  # UUID строки
                data={"main_text": main_text, "date": date, "page_schema_id": page_schema_id},
            )

            return True
        except Exception as e:
            print(f"❌ Error updating parsed content: {e}")
            return False

    def get_search_results(self, limit: Optional[int] = None):
        """Get search results not yet present in parsed_content."""
        search_results = self.db.select("search_results")
        parsed_results = self.db.select("parsed_content", filters={"status": None})

        parsed_ids = set(parsed_results["search_result_id"].tolist()) if not parsed_results.empty else set()

        filtered = search_results[~search_results["id"].isin(parsed_ids)]

        return list(filtered.head(limit).to_dict("records")) if limit else list(filtered.to_dict("records"))

    def parse_record(
        self,
        record: dict,
        force_reextract: bool = False,
    ) -> bool:
        """Parse a single record from search_results.

        Args:
            record: Dict with keys: id, link, title, html_storage_path.
            force_reextract: Force schema re-extraction even if exists.

        Returns:
            True if successful, False otherwise.
        """
        search_result_id = str(record["id"])
        url = record["link"]
        title = record["title"]
        html_storage_path = record.get("html_storage_path")

        parsed_content_id = self._save_parsed_content(
            search_result_id=search_result_id,
            main_text=[],
            date=None,
            page_schema_id=None,
        )

        domain = self._get_domain(url)

        print(f"\n{'='*100}")
        print(f"🌐 Processing: {title}")
        print(f"🔗 URL: {url}")
        print(f"📍 Domain: {domain}")
        print(f"📁 Storage path: {html_storage_path}")

        # Step 1: Download HTML from storage
        if not html_storage_path:
            print(f"❌ No HTML storage path for this record")
            self.mark_as(parsed_content_id, "failed")
            return False

        html = self.file_storage.download(html_storage_path)
        if not html:
            print(f"❌ Failed to download HTML from storage")
            self.mark_as(parsed_content_id, "failed")
            return False

        print(f"✅ HTML downloaded ({len(html)} characters)")

        # Step 2: Check if schema exists
        schema = self.schema_manager.get_by_domain(domain)

        if not schema or force_reextract:
            print(f"📝 No schema found for {domain}, extracting...")
            schema = self.schema_extractor.extract_schema(html, title, domain)

            if not schema:
                print(f"❌ Failed to extract schema for {domain}")
                self.mark_as(parsed_content_id, "failed")
                return False

            # Save schema
            schema_id = self.schema_manager.save(schema)
            if not schema_id:
                print(f"❌ Failed to save schema for {domain}")
                self.mark_as(parsed_content_id, "failed")
                return False

            schema.id = schema_id
            print(f"✅ Schema saved with ID: {schema_id}")
        else:
            print(f"✅ Using existing schema (ID: {schema.id})")

        # Check if page is valid
        if not schema.is_valid:
            print(f"⚠️ Page is not about earthquakes, skipping...")
            self.mark_as(parsed_content_id, "failed")
            return False

        # Step 3: Extract data
        result = self.data_extractor.extract(html, schema)

        # Step 4: Check if extraction was successful
        # Failed only if main_text is empty (date can be None)
        if not result.main_text:
            print(f"⚠️ Extraction failed (main_text empty), re-extracting schema...")

            # Re-extract schema
            schema = self.schema_extractor.extract_schema(html, title, domain)
            if not schema:
                print(f"❌ Failed to re-extract schema")
                self.mark_as(parsed_content_id, "failed")
                return False

            # Save updated schema
            schema_id = self.schema_manager.save(schema)
            if not schema_id:
                print(f"❌ Failed to save re-extracted schema")
                self.mark_as(parsed_content_id, "failed")
                return False

            schema.id = schema_id

            # Try extraction again
            result = self.data_extractor.extract(html, schema)

            if not result.main_text:
                print(f"❌ Re-extraction also failed, marking as failed...")
                self.mark_as(parsed_content_id, "failed")
                return False

        # Step 5: Save parsed content
        print(f"📌 Extracted text: {len(result.main_text)} paragraphs")
        print(f"📅 Date: {result.date or 'Not found (OK)'}")

        content_id = self.update_parsed_content(
            parsed_content_id=parsed_content_id,
            main_text=result.main_text,
            date=result.date,
            page_schema_id=schema.id,
        )

        if content_id:
            print(f"✅ Content saved with ID: {content_id}")
            self.mark_as(parsed_content_id, "parsed")
            return True
        else:
            print(f"❌ Failed to save content")
            self.mark_as(parsed_content_id, "failed")
            return False

    def parse_downloaded(self, limit: Optional[int] = 100) -> dict:
        """Parse all downloaded URLs from search_results table.

        Args:
            limit: Maximum number of records to process.

        Returns:
            Dictionary with statistics.
        """
        print(f"📥 Loading downloaded search results from database...")

        # Get records with status='downloaded'
        records = self.get_search_results(limit=limit)

        if not records:
            print("⚠️ No downloaded search results found in database")
            return {"total": 0, "successful": 0, "failed": 0}

        print(f"✅ Loaded {len(records)} downloaded results")

        stats = {
            "total": len(records),
            "successful": 0,
            "failed": 0,
        }

        for record in records:
            try:
                success = self.parse_record(record)

                if success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                print(f"❌ Error processing record: {e}")
                stats["failed"] += 1

                # Mark as failed in DB
                try:
                    self.mark_as(str(record["id"]), "failed")
                except:
                    pass

        print(f"\n{'='*100}")
        print(f"📊 Parsing complete:")
        print(f"   Total: {stats['total']}")
        print(f"   ✅ Successful: {stats['successful']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"{'='*100}")

        return stats

    def mark_as(self, parsed_content_id: str, status: str) -> bool:
        """Mark a search result as downloaded.

        Args:
            parsed_content_id: ID of the parsed result.
            status: Search status (e.g. 'parsed').

        Returns:
            True if successful, False otherwise.
        """
        updated = self.db.update(
            "parsed_content",
            parsed_content_id,
            {"status": status},
        )
        return updated is not None
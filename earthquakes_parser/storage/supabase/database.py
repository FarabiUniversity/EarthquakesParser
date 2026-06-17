"""Supabase database utility - low-level database operations."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _load_local_env() -> None:
    """Load Supabase env vars from common repo-local `.env` locations."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[3]
    for env_path in (repo_root / ".env", repo_root / "veritatis" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_local_env()


class SupabaseDB:
    """Low-level Supabase PostgreSQL database operations.

    This is a utility class providing generic CRUD operations.
    Business logic should be in domain modules (parser, searcher).
    """

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase database client.

        Args:
            url: Supabase project URL. Defaults to SUPABASE_URL env var.
            key: Supabase service role key. Defaults to SUPABASE_KEY env var.

        Raises:
            ImportError: If supabase package is not installed.
            ValueError: If URL or key is missing.
        """
        try:
            from supabase import create_client  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "supabase package is required. Install with: pip install supabase"
            ) from e

        # Get credentials
        self.url = url or os.getenv("SUPABASE_URL")
        # Prefer service role key when present for backend/server usage.
        self.key = (
            key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        )

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key are required. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables "
                "or pass them explicitly."
            )

        # Initialize client
        self.client = create_client(self.url, self.key)

    def insert(
        self, table: str, data: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[str]:
        """Insert records into table.

        Args:
            table: Table name.
            data: List of records to insert.
            batch_size: Number of records per batch.

        Returns:
            List of inserted record IDs.
        """
        inserted_ids = []

        try:
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                response = self.client.table(table).insert(batch).execute()

                if response.data:
                    batch_ids = [str(record["id"]) for record in response.data]
                    inserted_ids.extend(batch_ids)

            return inserted_ids

        except Exception as e:
            print(f"Error inserting into {table}: {e}")
            return inserted_ids

    def upsert(
        self,
        table: str,
        data: List[Dict[str, Any]],
        on_conflict: str,
        batch_size: int = 100,
    ) -> List[str]:
        """Insert or update records in a table.

        Args:
            table: Table name.
            data: List of records to upsert.
            on_conflict: Comma-separated unique column list for conflict resolution.
            batch_size: Number of records per batch.

        Returns:
            List of upserted record IDs.
        """
        upserted_ids: List[str] = []

        try:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                response = (
                    self.client.table(table)
                    .upsert(batch, on_conflict=on_conflict)
                    .execute()
                )

                if response.data:
                    batch_ids = [str(record["id"]) for record in response.data]
                    upserted_ids.extend(batch_ids)

            return upserted_ids

        except Exception as e:
            print(f"Error upserting into {table}: {e}")
            return upserted_ids

    def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> "pd.DataFrame":
        """Select records from table.

        Args:
            table: Table name.
            columns: Columns to select (default: "*").
            filters: Dict of column: value filters (uses eq operator).
            limit: Maximum number of records.

        Returns:
            DataFrame with results.
        """
        try:
            query = self.client.table(table).select(columns)

            # Apply filters
            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)

            # Apply limit
            if limit:
                query = query.limit(limit)

            response = query.execute()
            import pandas as pd

            return pd.DataFrame(response.data)

        except Exception as e:
            print(f"Error selecting from {table}: {e}")
            import pandas as pd

            return pd.DataFrame()

    def update(
        self, table: str, record_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a record by ID.

        Args:
            table: Table name.
            record_id: Record ID.
            data: Fields to update.

        Returns:
            Updated record or None if failed.
        """
        try:
            response = (
                self.client.table(table).update(data).eq("id", record_id).execute()
            )

            if response.data:
                return dict(response.data[0])
            return None

        except Exception as e:
            print(f"Error updating {table}: {e}")
            return None

    def delete(self, table: str, record_id: str) -> bool:
        """Delete a record by ID.

        Args:
            table: Table name.
            record_id: Record ID.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.table(table).delete().eq("id", record_id).execute()
            return True

        except Exception as e:
            print(f"Error deleting from {table}: {e}")
            return False

    def exists(self, table: str, column: str, value: Any) -> bool:
        """Check if record exists.

        Args:
            table: Table name.
            column: Column to check.
            value: Value to match.

        Returns:
            True if exists, False otherwise.
        """
        try:
            response = (
                self.client.table(table)
                .select("id")
                .eq(column, value)
                .limit(1)
                .execute()
            )

            return len(response.data) > 0

        except Exception as e:
            print(f"Error checking existence in {table}: {e}")
            return False

    def get_by_id(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID.

        Args:
            table: Table name.
            record_id: Record ID.

        Returns:
            Record data or None if not found.
        """
        try:
            response = (
                self.client.table(table).select("*").eq("id", record_id).execute()
            )

            if response.data:
                return dict(response.data[0])
            return None

        except Exception as e:
            print(f"Error getting record from {table}: {e}")
            return None

    def execute_sql(self, query: str) -> "pd.DataFrame":
        """Execute raw SQL query.

        Args:
            query: SQL query string.

        Returns:
            DataFrame with results.
        """
        try:
            response = self.client.rpc("execute_sql", {"query": query}).execute()
            import pandas as pd

            return pd.DataFrame(response.data)

        except Exception as e:
            print(f"Error executing SQL: {e}")
            import pandas as pd

            return pd.DataFrame()

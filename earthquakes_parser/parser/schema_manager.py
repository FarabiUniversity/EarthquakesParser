"""Schema management in database."""

import json
from typing import Optional

from earthquakes_parser.parser.models import PageSchema
from earthquakes_parser.storage.supabase.database import SupabaseDB


class SchemaManager:
    """Manages page schemas in database."""

    def __init__(self, db: SupabaseDB):
        """Initialize schema manager.

        Args:
            db: Database instance.
        """
        self.db = db
        self.table = "page_schemas"

    def get_by_domain(self, domain: str) -> Optional[PageSchema]:
        """Get schema by domain.

        Args:
            domain: Domain name (e.g., 'example.com').

        Returns:
            PageSchema or None if not found.
        """
        try:
            df = self.db.select(
                self.table,
                filters={"domain": domain},
                limit=1,
            )

            if df.empty:
                return None

            row = df.iloc[0]
            return PageSchema(
                id=str(row["id"]),
                domain=row["domain"],
                main_text_selectors=json.loads(row["main_text_selectors"]),
                date_selector=row.get("date_selector"),
                is_valid=row["is_valid"],
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
        except Exception as e:
            print(f"❌ Error getting schema for domain {domain}: {e}")
            return None

    def save(self, schema: PageSchema) -> Optional[str]:
        """Save or update schema.

        Args:
            schema: PageSchema to save.

        Returns:
            Schema ID or None if failed.
        """
        try:
            # Check if schema exists
            existing = self.get_by_domain(schema.domain)

            if existing:
                # Update existing
                schema_id = existing.id if existing.id else ""
                updated = self.db.update(
                    self.table,
                    schema_id,
                    schema.to_dict(),
                )
                return schema_id if updated else None
            else:
                # Insert new
                inserted_ids = self.db.insert(
                    self.table,
                    [schema.to_dict()],
                    batch_size=1,
                )
                return inserted_ids[0] if inserted_ids else None

        except Exception as e:
            print(f"❌ Error saving schema for domain {schema.domain}: {e}")
            return None

    def delete(self, schema_id: str) -> bool:
        """Delete schema by ID.

        Args:
            schema_id: Schema ID.

        Returns:
            True if successful.
        """
        return self.db.delete(self.table, schema_id)

    def exists(self, domain: str) -> bool:
        """Check if schema exists for domain.

        Args:
            domain: Domain name.

        Returns:
            True if exists.
        """
        return self.db.exists(self.table, "domain", domain)

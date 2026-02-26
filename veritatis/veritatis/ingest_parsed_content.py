"""Pull parsed content from Supabase and ingest into Veritatis (Milvus).

This script reads rows from the Supabase ``parsed_content`` table where
``status='parsed'``, normalises their ``main_text``, generates an embedding,
and inserts a single record per row into the unified ``veritatis`` Milvus
collection at **tier 1**.  The text itself is *not* stored in Milvus.

After successful ingestion the Supabase rows are marked ``status='ingested'``.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence, cast

from postgrest.exceptions import APIError

from veritatis.ingestion import ingest_record

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Prefer service role key for backend ingestion, but allow a generic SUPABASE_KEY
# for convenience (may fail if RLS blocks access).
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

_SUPABASE_CLIENT: Any = None


def _get_supabase_client():
    """Return a cached Supabase client.

    Prefers reusing the shared connector from ``earthquakes_parser`` when importable.
    Falls back to direct ``supabase.create_client`` to keep this script runnable in
    a standalone Veritatis environment.
    """
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Missing SUPABASE_URL and Supabase key "
            "(SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY)"
        )

    try:
        from earthquakes_parser.storage.supabase import SupabaseDB

        _SUPABASE_CLIENT = SupabaseDB(url=SUPABASE_URL, key=SUPABASE_KEY).client
        return _SUPABASE_CLIENT
    except Exception:
        from supabase import create_client  # type: ignore[attr-defined]

        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _SUPABASE_CLIENT


BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.2  # seconds


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _parse_date_to_epoch_ms(date_value: Any) -> int:
    """Convert a Supabase timestamptz string to epoch milliseconds.

    Returns 0 when the value is missing or unparseable.
    """
    if not date_value:
        return 0
    if isinstance(date_value, (int, float)):
        return int(date_value)
    try:
        dt = datetime.fromisoformat(str(date_value))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Text normalisation (kept from previous version, minus chunking)
# ---------------------------------------------------------------------------


def normalize_main_text(value: Any) -> str:
    """Convert Supabase main_text into a plain string suitable for embeddings.

    Some rows may contain JSON strings like ``"[]"`` or ``'["..."]'``.  Others
    may come through already decoded as list/dict depending on how they were
    stored.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                decoded = json.loads(s)
                return normalize_main_text(decoded)
            except Exception:
                return s
        return s

    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            item_s = normalize_main_text(item)
            if item_s:
                parts.append(item_s)
        return "\n".join(parts)

    if isinstance(value, dict):
        for key in ("text", "main_text", "content"):
            if key in value:
                return normalize_main_text(value.get(key))
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    return str(value)


# ---------------------------------------------------------------------------
# Supabase data fetching
# ---------------------------------------------------------------------------


def _extract_domain(row: Dict[str, Any]) -> str:
    """Extract the ``domain`` string from a joined ``page_schemas`` object."""
    ps = row.get("page_schemas")
    if isinstance(ps, dict):
        return str(ps.get("domain", ""))
    return ""


def fetch_parsed_content(
    offset: int,
    limit: int,
    status: str = "parsed",
) -> List[Dict[str, Any]]:
    """Fetch a page of rows from Supabase ``parsed_content`` with joins.

    Joins ``page_schemas`` (via ``page_schema_id``) to retrieve the domain.
    Ordering prefers ``date``; falls back to ``id`` if the column is absent.
    """
    client = _get_supabase_client()

    select_cols = "id, main_text, date, page_schema_id, page_schemas(domain)"

    try:
        response = (
            client.table("parsed_content")
            .select(select_cols)
            .eq("status", status)
            .order("date", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return cast(List[Dict[str, Any]], response.data)
    except APIError as e:
        msg = str(e)
        if "date" not in msg:
            raise
        # Fallback: order by id if ``date`` column causes issues.
        response = (
            client.table("parsed_content")
            .select(select_cols)
            .eq("status", status)
            .order("id", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return cast(List[Dict[str, Any]], response.data)


def _unique_ids(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def mark_ingested(parsed_content_ids: Sequence[str]) -> None:
    """Mark parsed_content rows as ingested in Supabase."""
    if not parsed_content_ids:
        return

    client = _get_supabase_client()
    try:
        client.table("parsed_content").update({"status": "ingested"}).in_(
            "id", list(parsed_content_ids)
        ).execute()
    except APIError as e:
        msg = str(e)
        if "invalid input value for enum" in msg and "processing_status" in msg:
            raise RuntimeError(
                "Supabase enum 'processing_status' does not include 'ingested'. "
                "Apply the migration (or run the ALTER TYPE) and retry."
            ) from e
        raise


# ---------------------------------------------------------------------------
# Ingestion orchestration
# ---------------------------------------------------------------------------


def send_to_ingest(records: List[Dict[str, Any]]) -> List[str]:
    """Ingest Supabase rows into Veritatis (Milvus) directly.

    One Milvus record per ``parsed_content`` row — no chunking.
    Returns a deduplicated list of parsed_content IDs that were processed.
    """
    ingested_parsed_content_ids: List[str] = []

    for r in records:
        text = normalize_main_text(r.get("main_text"))
        if not text:
            continue

        parsed_content_id = r["id"]
        date_ms = _parse_date_to_epoch_ms(r.get("date"))
        domain = _extract_domain(r)

        ingest_record(
            text,
            iid=parsed_content_id,
            tier=1,
            credibility_score=0.0,
            date=date_ms,
            domain=domain,
        )

        ingested_parsed_content_ids.append(parsed_content_id)

    return _unique_ids(ingested_parsed_content_ids)


def main():
    """CLI entry point for batch ingestion."""
    while True:
        batch = fetch_parsed_content(0, BATCH_SIZE)

        if not batch:
            print("✅ No more records to ingest.")
            break

        print(f"➡️ Ingesting batch ({len(batch)} records)")
        ingested_ids = send_to_ingest(batch)
        if ingested_ids:
            mark_ingested(ingested_ids)
        time.sleep(SLEEP_BETWEEN_BATCHES)


if __name__ == "__main__":
    main()

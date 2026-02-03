"""Pull parsed content from Supabase and ingest into the Veritatis API.

This script reads rows from the Supabase `parsed_content` table where
`status='parsed'`, normalizes and chunks their `main_text`, POSTs tier-1 records
to the local `/ingest` API, and then marks those rows as `status='ingested'`.
"""

import json
import os
import time
from typing import Any, Dict, Iterable, List, Sequence, cast

import requests
from postgrest.exceptions import APIError

from supabase import create_client  # type: ignore[attr-defined]

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Optional: the script still works if env vars are set externally.
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Prefer service role key for backend ingestion, but allow a generic SUPABASE_KEY
# for convenience (may fail if RLS blocks access).
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# Default to the local FastAPI port used in this repo.
# In docker-compose.yml the API is exposed on 8000.
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8000/ingest")

BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.2  # seconds

# Milvus VARCHAR limits are byte-based; keep a safety margin.
# NOTE: Veritatis Milvus schema uses FieldSchema(name="content", max_length=10000)
# which is enforced as a character length check. Keep a safety margin.
MAX_TEXT_CHARS = 9500
# Milvus VARCHAR limits are effectively UTF-8 byte-based; keep a safety margin.
MAX_TEXT_BYTES = 9500


def _utf8_len(s: str) -> int:
    """Return the UTF-8 byte length of a string."""
    return len(s.encode("utf-8"))


def chunk_text(
    text: str,
    max_bytes: int = MAX_TEXT_BYTES,
    max_chars: int = MAX_TEXT_CHARS,
) -> List[str]:
    """Split text into chunks that fit both byte and character limits."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars and _utf8_len(text) <= max_bytes:
        return [text]

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_bytes = 0
    current_chars = 0

    for w in words:
        w_bytes = _utf8_len(w)
        w_chars = len(w)

        # If a single token is larger than either limit, split it by characters.
        if w_bytes > max_bytes or w_chars > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_bytes = 0
                current_chars = 0

            buf: List[str] = []
            buf_bytes = 0
            buf_chars = 0
            for ch in w:
                ch_b = _utf8_len(ch)
                if buf and (buf_bytes + ch_b > max_bytes or buf_chars + 1 > max_chars):
                    chunks.append("".join(buf))
                    buf = [ch]
                    buf_bytes = ch_b
                    buf_chars = 1
                else:
                    buf.append(ch)
                    buf_bytes += ch_b
                    buf_chars += 1
            if buf:
                chunks.append("".join(buf))
            continue

        sep_bytes = 1 if current else 0  # space
        sep_chars = 1 if current else 0
        if (
            current_bytes + sep_bytes + w_bytes > max_bytes
            or current_chars + sep_chars + w_chars > max_chars
        ):
            chunks.append(" ".join(current))
            current = [w]
            current_bytes = w_bytes
            current_chars = w_chars
        else:
            current.append(w)
            current_bytes += sep_bytes + w_bytes
            current_chars += sep_chars + w_chars

    if current:
        chunks.append(" ".join(current))

    return chunks


def normalize_main_text(value) -> str:
    """Convert Supabase main_text into a plain string suitable for embeddings.

    Some rows may contain JSON strings like "[]" or '["..."]'. Others may
    come through already decoded as list/dict depending on how they were stored.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        # If it looks like JSON, try to parse it.
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
        # Common patterns: {"text": "..."} or {"main_text": "..."}
        for key in ("text", "main_text", "content"):
            if key in value:
                return normalize_main_text(value.get(key))
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    return str(value)


def fetch_parsed_content(offset: int, limit: int) -> List[Dict[str, Any]]:
    """Fetch a page of rows from Supabase `parsed_content`.

    Tries ordering by `parsed_at` if present; falls back to ordering by `id`.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Missing SUPABASE_URL and Supabase key "
            "(SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY)"
        )

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Try to use parsed_at ordering if the column exists; fall back to id ordering.
    try:
        response = (
            client.table("parsed_content")
            .select("id, main_text, search_result_id, parsed_at")
            .eq("status", "parsed")
            .order("parsed_at", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return cast(List[Dict[str, Any]], response.data)
    except APIError as e:
        msg = str(e)
        if "parsed_at" not in msg:
            raise
        response = (
            client.table("parsed_content")
            .select("id, main_text, search_result_id")
            .eq("status", "parsed")
            .order("id", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return cast(List[Dict[str, Any]], response.data)


def _unique_ids(values: Iterable[str]) -> List[str]:
    seen = set()
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
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Missing SUPABASE_URL and Supabase key "
            "(SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY)"
        )

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
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


def send_to_ingest(records: List[Dict[str, Any]]) -> List[str]:
    """Ingest Supabase rows into Veritatis.

    Note: the Veritatis API in this repo accepts one record per request at
    POST /ingest with body fields: content, source_url, supabase_id, metadata.
    """
    ingested_parsed_content_ids: List[str] = []

    for r in records:
        text = normalize_main_text(r.get("main_text"))
        if not text:
            # Skip empty/NULL content to avoid 422s.
            continue

        parent_id = r["id"]
        chunks = chunk_text(text)
        if not chunks:
            continue

        ingested_parsed_content_ids.append(parent_id)

        base_metadata = {
            "search_result_id": r.get("search_result_id"),
            **({"parsed_at": r.get("parsed_at")} if r.get("parsed_at") else {}),
        }

        # Avoid sending null metadata keys.
        base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

        for idx, chunk in enumerate(chunks):
            body = {
                "content": chunk,
                "supabase_id": parent_id,
                "metadata": {
                    **base_metadata,
                    "parent_id": parent_id,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                },
            }

            resp = requests.post(INGEST_URL, json=body, timeout=60)
            if not resp.ok:
                raise RuntimeError(f"Ingest failed ({resp.status_code}): {resp.text}")

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

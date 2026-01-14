"""Pull parsed content from Supabase and ingest into the Veritatis API.

This script reads rows from the Supabase `parsed_content` table, normalizes and
chunks their `main_text`, and POSTs tier-1 records to the local `/ingest` API.
"""

import json
import os
import time
from typing import Any, Dict, List, cast

import requests
from postgrest.exceptions import APIError

from supabase import create_client  # type: ignore[attr-defined]

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Prefer service role key for backend ingestion, but allow a generic SUPABASE_KEY
# for convenience (may fail if RLS blocks access).
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# Default to the local FastAPI port used in this repo (see uvicorn command in README).
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8001/ingest")

BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.2  # seconds

# Milvus VARCHAR limits are byte-based; keep a safety margin.
MAX_TEXT_BYTES = 60000


def _utf8_len(s: str) -> int:
    """Return the UTF-8 byte length of a string."""
    return len(s.encode("utf-8"))


def chunk_text(text: str, max_bytes: int = MAX_TEXT_BYTES) -> List[str]:
    """Split text into chunks that are each <= `max_bytes` when UTF-8 encoded."""
    text = (text or "").strip()
    if not text:
        return []
    if _utf8_len(text) <= max_bytes:
        return [text]

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_bytes = 0

    for w in words:
        w_bytes = _utf8_len(w)

        # If a single token is larger than max_bytes, split it by characters.
        if w_bytes > max_bytes:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_bytes = 0

            buf: List[str] = []
            buf_bytes = 0
            for ch in w:
                ch_b = _utf8_len(ch)
                if buf and buf_bytes + ch_b > max_bytes:
                    chunks.append("".join(buf))
                    buf = [ch]
                    buf_bytes = ch_b
                else:
                    buf.append(ch)
                    buf_bytes += ch_b
            if buf:
                chunks.append("".join(buf))
            continue

        sep_bytes = 1 if current else 0  # space
        if current_bytes + sep_bytes + w_bytes > max_bytes:
            chunks.append(" ".join(current))
            current = [w]
            current_bytes = w_bytes
        else:
            current.append(w)
            current_bytes += sep_bytes + w_bytes

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
            .order("id", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return cast(List[Dict[str, Any]], response.data)


def send_to_ingest(records: List[Dict[str, Any]]):
    """Transform Supabase rows into tier-1 records and POST them to `/ingest`."""
    payload = []

    for r in records:
        text = normalize_main_text(r.get("main_text"))
        if not text:
            # Skip empty/NULL content to avoid 422s.
            continue

        parent_id = r["id"]
        chunks = chunk_text(text)
        if not chunks:
            continue

        base_metadata = {
            "search_result_id": r.get("search_result_id"),
            **({"parsed_at": r.get("parsed_at")} if r.get("parsed_at") else {}),
        }

        # Avoid sending null metadata keys.
        base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

        for idx, chunk in enumerate(chunks):
            chunk_id = parent_id if len(chunks) == 1 else f"{parent_id}:{idx}"
            payload.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        **base_metadata,
                        "parent_id": parent_id,
                        "chunk_index": idx,
                        "chunk_count": len(chunks),
                    },
                }
            )

    if not payload:
        return

    resp = requests.post(INGEST_URL, json=payload, timeout=60)

    if not resp.ok:
        raise RuntimeError(f"Ingest failed ({resp.status_code}): {resp.text}")


def main():
    """CLI entry point for batch ingestion."""
    offset = 0

    while True:
        batch = fetch_parsed_content(offset, BATCH_SIZE)

        if not batch:
            print("✅ No more records to ingest.")
            break

        print(f"➡️ Ingesting batch starting at offset {offset} ({len(batch)} records)")
        send_to_ingest(batch)

        offset += BATCH_SIZE
        time.sleep(SLEEP_BETWEEN_BATCHES)


if __name__ == "__main__":
    main()

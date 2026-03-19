"""Core ingestion logic for Veritatis.

Provides `ingest_record()` which normalizes text, generates an embedding,
and inserts into the Milvus ``veritatis_tier1`` collection.  The text itself
is **not** stored in Milvus — only the embedding vector and metadata
(credibility_score, date, domain).  The ``iid`` primary key maps back
to ``parsed_content.id`` in Supabase for full-text retrieval when needed.

This module is used by both the FastAPI endpoint and the batch ingestion scripts.
"""

from dataclasses import dataclass
from typing import Optional

from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import (
    ALL_TIER_COLLECTIONS,
    MilvusRecordStore,
    collection_for_tier,
    ensure_connection,
    init_collections,
)

_store: Optional[MilvusRecordStore] = None


@dataclass
class IngestResult:
    """Result of a single ingest operation."""

    iid: str
    status: str  # "inserted" | "duplicate"
    collection: str


def get_store() -> MilvusRecordStore:
    """Return the cached MilvusRecordStore, creating it on first call."""
    global _store
    if _store is not None:
        return _store
    ensure_connection()
    init_collections()
    _store = MilvusRecordStore()
    return _store


def set_store(store: MilvusRecordStore) -> None:
    """Override the cached store (useful for testing or API lifespan setup)."""
    global _store
    _store = store


def ingest_record(
    text: str,
    *,
    iid: str,
    tier: int = 1,
    credibility_score: float = 0.0,
    date: int = 0,
    domain: str = "",
    store: Optional[MilvusRecordStore] = None,
    flush: bool = True,
) -> IngestResult:
    """Ingest a single parsed_content record into the Milvus tier collection.

    The *text* is used only to generate the embedding — it is **not** stored
    in Milvus.  All other metadata fields are persisted alongside the vector.

    Parameters
    ----------
    text:
        Raw text (main_text) to generate the embedding from.
    iid:
        Primary key — must equal ``parsed_content.id`` (UUID).
    tier:
        Credibility tier (1 = raw/unverified, 2 = credible, 3 = verified).
        Determines which collection the record is inserted into.
    credibility_score:
        Initial credibility score (default 0.0, updated later).
    date:
        Earthquake event date as epoch milliseconds.
    domain:
        Source domain from ``page_schemas``.
    store:
        Optional MilvusRecordStore override; falls back to module-level singleton.

    Returns
    -------
    IngestResult with the iid, status ('inserted' or 'duplicate'),
    and collection name.
    """
    store = store or get_store()
    target_collection = collection_for_tier(tier)

    # Global dedup: prevent the same iid existing in multiple tier collections.
    existing_collection = store.find_record_collection(
        iid, collections=ALL_TIER_COLLECTIONS
    )
    if existing_collection:
        return IngestResult(iid=iid, status="duplicate", collection=existing_collection)

    # Generate embedding from text (text itself is NOT stored in Milvus)
    normalized = " ".join(text.split()).strip()
    embedding = embedding_generator.embed(normalized)

    record = {
        "iid": iid,
        "embedding": embedding,
        "credibility_score": credibility_score,
        "date": date,
        "domain": domain,
    }

    store.insert_record(target_collection, record, flush=flush)
    return IngestResult(iid=iid, status="inserted", collection=target_collection)

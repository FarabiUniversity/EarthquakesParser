"""Core ingestion logic for Veritatis.

Provides `ingest_record()` which normalizes text, generates an embedding,
and inserts into the single Milvus ``veritatis`` collection.  The text itself
is **not** stored in Milvus — only the embedding vector and metadata
(tier, credibility_score, date, domain).  The ``iid`` primary key maps back
to ``parsed_content.id`` in Supabase for full-text retrieval when needed.

This module is used by both the FastAPI endpoint and the batch ingestion scripts.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import (
    COLLECTION_NAME,
    MilvusRecordStore,
    ensure_connection,
    init_collections,
)

logger = logging.getLogger(__name__)

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
) -> IngestResult:
    """Ingest a single parsed_content record into Milvus.

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

    # Dedup: check if iid already exists
    if store.record_exists(COLLECTION_NAME, iid):
        return IngestResult(iid=iid, status="duplicate", collection=COLLECTION_NAME)

    # Generate embedding from text (text itself is NOT stored in Milvus)
    normalized = " ".join(text.split()).strip()
    embedding = embedding_generator.embed(normalized)

    record = {
        "iid": iid,
        "embedding": embedding,
        "tier": tier,
        "credibility_score": credibility_score,
        "date": date,
        "domain": domain,
    }

    store.insert_record(COLLECTION_NAME, record)
    return IngestResult(iid=iid, status="inserted", collection=COLLECTION_NAME)

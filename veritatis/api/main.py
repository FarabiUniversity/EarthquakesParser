"""Main entry point for the veritatis API using FastAPI."""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymilvus import Collection
from pymilvus.orm import utility

from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import (
    MilvusRecordStore,
    ensure_collection_loaded,
    ensure_connection,
    init_collections,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_TIER1 = "veritatis_tier1_lake"
_TIER2 = "veritatis_tier2_arena"
_store = None  # Initialize on startup

# Hoist FastAPI Body() defaults to module scope to satisfy flake8-bugbear (B008).
_BODY_INGEST_RECORDS = Body(...)
_BODY_QUERY = Body(..., embed=True)
_BODY_TOP_K = Body(10)
_BODY_MOVE_RECORD_ID = Body(..., embed=True)
_BODY_MOVE_VERIFICATION_CONFIDENCE = Body(0.0)
_BODY_MOVE_CROSS_SOURCE_COUNT = Body(1)


class IngestRecord(BaseModel):
    """Tier-1 ingest payload record with canonical fields: id, text, metadata."""

    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    global _store
    # Startup
    logger.info("Starting application initialization...")

    try:
        logger.info("Connecting to Milvus...")
        ensure_connection()
        logger.info("Milvus connection established")
        logger.info("Initializing collections...")
        init_collections()
        logger.info("Collections initialized")

        logger.info("Creating MilvusRecordStore...")
        _store = MilvusRecordStore()
        logger.info("Milvus connected and collections initialized")
    except Exception as e:
        logger.warning(f"Milvus not available: {e}")
        logger.warning("API will start but vector operations will fail")

    # Warm up embedding model (works without Milvus)
    logger.info(
        "Warming up embedding model (this may take 5-10 seconds on first run)..."
    )
    start_time = time.time()
    try:
        embedding_generator.embed("warmup")
        elapsed = time.time() - start_time
        logger.info(f"✅ Embedding model ready (took {elapsed:.2f}s)")
    except Exception as e:
        logger.error(f"❌ Embedding model error: {e}")
        raise

    logger.info("Application startup complete!")
    yield
    # Shutdown (if needed)
    logger.info("🛑 API shutting down")


app = FastAPI(title="Veritatis API", version="1.0", lifespan=lifespan)

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    """Return API health status."""
    return {"status": "ok"}


@app.get("/debug/collections")
async def list_collections():
    """List existing Milvus collections (debug endpoint)."""
    try:
        ensure_connection()
        cols = utility.list_collections()
        return {"collections": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus error: {e}")


# --- Ingestion endpoint ---
@app.post("/ingest")
async def ingest(records: list[IngestRecord] = _BODY_INGEST_RECORDS):
    """Ingest one or more text records into Tier1 with embeddings.

    Expected payload is a JSON array of objects with tier-1 fields:
    - id: string
    - text: string
    - metadata: object

    If an id already exists in Tier1 or Tier2, ingestion is rejected with HTTP 409.
    """
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Check Milvus connection.",
        )

    if not records:
        raise HTTPException(
            status_code=422, detail="Request body must be a non-empty list"
        )

    # Dedup check (across Tier 1 and Tier 2)
    try:
        ensure_collection_loaded(_TIER1)
        ensure_collection_loaded(_TIER2)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Milvus not ready: {e}")

    normalized_texts: list[str] = []
    ids: list[str] = []
    metadata_json: list[str] = []

    for r in records:
        rid = r.id.strip()
        if not rid:
            raise HTTPException(
                status_code=422, detail="Each record must have a non-empty 'id'"
            )
        if _store.record_exists(_TIER1, rid) or _store.record_exists(_TIER2, rid):
            raise HTTPException(status_code=409, detail=f"Duplicate record id: {rid}")

        normalized = " ".join(r.text.split()).strip()
        if not normalized:
            raise HTTPException(
                status_code=422, detail=f"Record '{rid}' has empty 'text'"
            )

        ids.append(rid)
        normalized_texts.append(normalized)
        metadata_json.append(
            json.dumps(r.metadata or {}, ensure_ascii=False, sort_keys=True)
        )

    embeddings = (
        embedding_generator.embed_batch(normalized_texts)
        if len(normalized_texts) > 1
        else [embedding_generator.embed(normalized_texts[0])]
    )

    rows: list[Dict[str, Any]] = []
    for i, rid in enumerate(ids):
        rows.append(
            {
                "id": rid,
                "text": normalized_texts[i],
                "embedding": embeddings[i],
                "metadata": metadata_json[i],
            }
        )

    try:
        _store.insert_record(_TIER1, rows)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"status": "inserted", "count": len(ids), "ids": ids, "collection": _TIER1}


# --- Vector search endpoint ---
@app.post("/search")
async def search(
    query: str = _BODY_QUERY,
    top_k: int = _BODY_TOP_K,
):
    """Run a vector similarity search against Tier1."""
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Check Milvus connection.",
        )

    ensure_collection_loaded(_TIER1)
    vec = embedding_generator.embed(query)
    collection = Collection(_TIER1)
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    try:
        results = collection.search(
            data=[vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id", "text", "metadata"],
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # results is a list per query (we only have one)
    hits = []
    for hit in results[0]:
        md_raw = str(getattr(hit, "metadata", ""))
        try:
            md = json.loads(md_raw) if md_raw else {}
        except Exception:
            md = {"_raw": md_raw}

        hits.append(
            {
                "id": str(hit.id),
                "text": str(getattr(hit, "text", "")),
                "metadata": md,
                "distance": float(hit.distance),
            }
        )

    return {"query": query, "top_k": top_k, "results": hits}


# --- Move/promote endpoint ---
@app.post("/move")
async def move_tier1_to_tier2(
    record_id: str = _BODY_MOVE_RECORD_ID,
    verification_confidence: float = _BODY_MOVE_VERIFICATION_CONFIDENCE,
    cross_source_count: int = _BODY_MOVE_CROSS_SOURCE_COUNT,
):
    """Move a record from Tier1 (lake) to Tier2 (arena) by id.

    Tier2 has a different schema than Tier1, so we project the Tier1 record into
    Tier2 fields and drop Tier1-only fields.
    """
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Check Milvus connection.",
        )

    if (
        not isinstance(cross_source_count, int)
        or isinstance(cross_source_count, bool)
        or cross_source_count < 0
    ):
        raise HTTPException(
            status_code=422, detail="cross_source_count must be a non-negative integer"
        )

    ensure_collection_loaded(_TIER1)
    ensure_collection_loaded(_TIER2)

    # Ensure the source record exists
    src = _store.get_record(_TIER1, record_id)
    if not src:
        raise HTTPException(
            status_code=404, detail=f"Record not found in {_TIER1}: {record_id}"
        )

    # Avoid accidental overwrite / inconsistent move
    if _store.record_exists(_TIER2, record_id):
        raise HTTPException(
            status_code=409, detail=f"Record already exists in {_TIER2}: {record_id}"
        )

    # Map Tier1 -> Tier2 schema
    tier2_record = {
        "id": record_id,
        "text": str(src.get("text", "")),
        "embedding": src.get("embedding"),
        "metadata": str(src.get("metadata", "")),
        "verification_confidence": float(verification_confidence),
        "cross_source_count": int(cross_source_count),
    }

    try:
        _store.insert_record(_TIER2, tier2_record)
        _store.delete_record(_TIER1, record_id)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"id": record_id, "status": "moved", "from": _TIER1, "to": _TIER2}


# --- Error handler example ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions and return a JSON error response."""
    return JSONResponse(
        status_code=500, content={"message": f"Unexpected error: {exc}"}
    )

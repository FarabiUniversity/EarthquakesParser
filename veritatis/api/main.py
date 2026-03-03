"""Main entry point for the veritatis API."""

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymilvus.orm import utility  # noqa: E402

from veritatis.embeddings import embedding_generator  # noqa: E402
from veritatis.vector_consensus import find_most_relevant_vector  # noqa: E402
from veritatis.vector_stores import (  # noqa: E402
    MilvusRecordStore,
    ensure_connection,
    init_collections,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_TIER1 = "veritatis_tier1_lake"
_store = None  # Initialize on startup


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
async def ingest(
    content: str = Body(..., embed=True),  # noqa: B008
    source_url: Optional[str] = Body(default=None),  # noqa: B008
    supabase_id: Optional[str] = Body(default=None),  # noqa: B008
    metadata: Optional[Dict[str, Any]] = Body(default=None),  # noqa: B008
):
    """Ingest a text record into Tier1 with embeddings and dedup by hash."""
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Check Milvus connection.",
        )

    normalized = " ".join(content.split()).strip()
    base = normalized + ("|" + source_url if source_url else "")
    record_id = hashlib.sha256(base.encode("utf-8")).hexdigest()

    # Dedup check
    if _store.record_exists(_TIER1, record_id):
        return {"id": record_id, "status": "duplicate", "collection": _TIER1}

    embedding = embedding_generator.embed(normalized)
    now_ms = int(time.time() * 1000)

    record = {
        "id": record_id,
        "content": normalized,
        "embedding": embedding,
        "source_url": source_url or "",
        "credibility_score": 0.0,
        "ingested_timestamp": now_ms,
        "supabase_id": supabase_id or "",
    }

    try:
        _store.insert_record(_TIER1, record)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"id": record_id, "status": "inserted", "collection": _TIER1}


_TIER2 = "veritatis_tier2_arena"


# --- Consensus analysis endpoint ---
@app.post("/consensus/analyze")
async def consensus_analyze(
    limit: Optional[int] = Body(default=None),  # noqa: B008
    top_n: int = Body(default=10),  # noqa: B008
    threshold: float = Body(default=0.7),  # noqa: B008
    centrality_weight: float = Body(default=0.6),  # noqa: B008
    detail_weight: float = Body(default=0.4),  # noqa: B008
    credibility_weight: float = Body(default=0.0),  # noqa: B008
):
    """Analyze Tier 1 vectors and promote the best to Tier 2.

    Compares all vectors against each other (no query needed) using centrality,
    detail, and credibility scores. The vector with the highest combined_score
    is always moved to Tier 2.

    Parameters:
    - limit: max number of vectors to analyze (None = all)
    - top_n: how many top results to include in the response
    - threshold: score threshold used only for reporting (above_threshold_count)
    - centrality_weight: weight for centrality score (default 0.6)
    - detail_weight: weight for detail score (default 0.4)
    - credibility_weight: weight for credibility score (default 0.0)

    All three weights must sum to 1.0.
    """
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Check Milvus connection.",
        )

    if abs(centrality_weight + detail_weight + credibility_weight - 1.0) > 1e-6:
        raise HTTPException(
            status_code=422,
            detail=(
                "centrality_weight + detail_weight + credibility_weight "
                "must equal 1.0, got "
                f"{centrality_weight + detail_weight + credibility_weight:.6f}"
            ),
        )

    try:
        best, all_ranked = find_most_relevant_vector(
            collection_name=_TIER1,
            centrality_weight=centrality_weight,
            detail_weight=detail_weight,
            credibility_weight=credibility_weight,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Consensus analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Move best vector to Tier 2 (schema mapping: Tier1 -> Tier2 fields)
    moved = False
    move_error: Optional[str] = None
    try:
        tier2_record = {
            "id": best.id,
            "content": best.content,
            "embedding": best.embedding,
            "verification_confidence": 0.0,
            "cross_source_count": 1,
            "supabase_id": best.supabase_id,
        }
        _store.insert_record(_TIER2, tier2_record)
        _store.delete_record(_TIER1, best.id)
        moved = True
        logger.info(f"Moved best vector {best.id} from Tier 1 to Tier 2")
    except Exception as e:
        move_error = str(e)
        logger.error(f"Failed to move vector {best.id} to Tier 2: {e}")

    # Build stats
    scores = [v.combined_score for v in all_ranked]
    centralities = [v.centrality_score for v in all_ranked]
    details = [v.detail_score for v in all_ranked]
    n = len(scores)

    top_list: List[Dict[str, Any]] = [
        {
            "rank": i + 1,
            "id": v.id,
            "combined_score": round(v.combined_score, 4),
            "centrality_score": round(v.centrality_score, 4),
            "detail_score": round(v.detail_score, 4),
            "content_length": v.content_length,
        }
        for i, v in enumerate(all_ranked[:top_n])
    ]

    response: Dict[str, Any] = {
        "analyzed_count": n,
        "best_vector": {
            "id": best.id,
            "content": best.content,
            "source_url": best.source_url,
            "credibility_score": best.credibility_score,
            "centrality_score": round(best.centrality_score, 4),
            "detail_score": round(best.detail_score, 4),
            "combined_score": round(best.combined_score, 4),
            "content_length": best.content_length,
        },
        "moved_to_tier2": moved,
        "top_n": top_list,
        "stats": {
            "avg_combined_score": round(sum(scores) / n, 4),
            "min_combined_score": round(min(scores), 4),
            "max_combined_score": round(max(scores), 4),
            "avg_centrality_score": round(sum(centralities) / n, 4),
            "avg_detail_score": round(sum(details) / n, 4),
            "above_threshold_count": sum(1 for s in scores if s >= threshold),
            "threshold_used": threshold,
        },
    }

    if move_error is not None:
        response["move_error"] = move_error

    return response


# --- Error handler example ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions and return a JSON error response."""
    return JSONResponse(
        status_code=500, content={"message": f"Unexpected error: {exc}"}
    )

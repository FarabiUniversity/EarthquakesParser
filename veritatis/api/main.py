"""Main entry point for the veritatis API."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymilvus.orm import utility  # noqa: E402

from veritatis.ingestion import IngestResult, ingest_record, set_store
from veritatis.vector_consensus import find_most_relevant_vector
from veritatis.vector_stores import (
    MilvusRecordStore,
    collection_for_tier,
    ensure_connection,
    init_collections,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module-level store reference (used by /move and /update_credibility)
_store: Optional[MilvusRecordStore] = None


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
        logger.info("Initializing collection...")
        init_collections()
        logger.info("Collection initialized")

        logger.info("Creating MilvusRecordStore...")
        _store = MilvusRecordStore()
        set_store(_store)
        logger.info("Milvus connected and collection initialized")
    except Exception as e:
        logger.warning(f"Milvus not available: {e}")
        logger.warning("API will start but vector operations will fail")

    # Warm up embedding model (works without Milvus)
    logger.info(
        "Warming up embedding model (this may take 5-10 seconds on first run)..."
    )
    start_time = time.time()
    try:
        from veritatis.embeddings import embedding_generator

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


app = FastAPI(title="Veritatis API", version="2.0", lifespan=lifespan)

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_store() -> MilvusRecordStore:
    """Return the module-level store, raising if unavailable."""
    if _store is None:
        raise HTTPException(status_code=503, detail="Milvus store not initialized")
    return _store


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
    text: str = Body(..., embed=True),  # noqa: B008
    iid: str = Body(..., embed=True),  # noqa: B008
    tier: int = Body(default=1),  # noqa: B008
    credibility_score: float = Body(default=0.0),  # noqa: B008
    date: int = Body(default=0),  # noqa: B008
    domain: str = Body(default=""),  # noqa: B008
):
    """Ingest a single record into the appropriate tier collection.

    The ``text`` is used to generate an embedding — it is NOT stored in Milvus.
    ``iid`` must be the ``parsed_content.id`` UUID from Supabase.
    ``tier`` selects the target collection (1 = raw, 2 = credible, 3 = verified).
    """
    try:
        result: IngestResult = ingest_record(
            text,
            iid=iid,
            tier=tier,
            credibility_score=credibility_score,
            date=date,
            domain=domain,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"iid": result.iid, "status": result.status, "collection": result.collection}


# ---------------------------------------------------------------------------
# Tier management endpoints
# ---------------------------------------------------------------------------


@app.post("/move")
async def move(
    iids: List[str] = Body(..., embed=True),  # noqa: B008
    source_tier: int = Body(..., embed=True),  # noqa: B008
    target_tier: int = Body(..., embed=True),  # noqa: B008
):
    """Move records between tier collections.

    Provide ``iids`` — a list of ``parsed_content.id`` UUIDs,
    ``source_tier`` (1-3), and ``target_tier`` (1-3).
    """
    store = _get_store()
    try:
        src = collection_for_tier(source_tier)
        tgt = collection_for_tier(target_tier)
        count = store.move_records(src, tgt, iids)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"moved": count, "source": src, "target": tgt}


@app.post("/update_credibility")
async def update_credibility(
    updates: List[Dict[str, Any]] = Body(..., embed=True),  # noqa: B008
    tier: int = Body(default=1),  # noqa: B008
):
    """Update credibility scores for one or more records.

    ``updates`` must be a list of objects, each with ``iid`` (str) and
    ``credibility_score`` (float).
    ``tier`` selects which collection to update (1, 2, or 3).
    """
    store = _get_store()
    try:
        col = collection_for_tier(tier)
        count = store.update_credibility_scores(col, updates)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"updated": count, "collection": col}


_TIER1 = "veritatis_tier1_lake"
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

    # Move best vector to Tier 2 (same schema across tiers)
    moved = False
    move_error: Optional[str] = None
    try:
        moved_count = _store.move_records(_TIER1, _TIER2, [best.iid])
        moved = moved_count == 1
        if moved:
            logger.info(f"Moved best vector {best.iid} from Tier 1 to Tier 2")
        else:
            move_error = "Record was not found in Tier 1 (already moved?)"
    except Exception as e:
        move_error = str(e)
        logger.error(f"Failed to move vector {best.iid} to Tier 2: {e}")

    # Build stats
    scores = [v.combined_score for v in all_ranked]
    centralities = [v.centrality_score for v in all_ranked]
    details = [v.detail_score for v in all_ranked]
    n = len(scores)

    top_list: List[Dict[str, Any]] = [
        {
            "rank": i + 1,
            "iid": v.iid,
            "combined_score": round(v.combined_score, 4),
            "centrality_score": round(v.centrality_score, 4),
            "detail_score": round(v.detail_score, 4),
            "main_text_length": v.main_text_length,
        }
        for i, v in enumerate(all_ranked[:top_n])
    ]

    response: Dict[str, Any] = {
        "analyzed_count": n,
        "best_vector": {
            "iid": best.iid,
            "main_text": best.main_text,
            "credibility_score": best.credibility_score,
            "date": best.date,
            "domain": best.domain,
            "centrality_score": round(best.centrality_score, 4),
            "detail_score": round(best.detail_score, 4),
            "combined_score": round(best.combined_score, 4),
            "main_text_length": best.main_text_length,
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

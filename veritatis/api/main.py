"""Main entry point for the veritatis API using FastAPI with relevance filtering."""

import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymilvus.orm import utility  # noqa: E402

from veritatis.agent import FactCheckResult, fact_check
from veritatis.ingestion import IngestResult, ingest_record, set_store
from veritatis.plain_search import vector_search
from veritatis.search import RelevanceFilter, search_with_relevance_filter
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


# --- Vector search endpoint (legacy - no filtering) ---
@app.post("/search")
async def search(
    query: str = Body(..., embed=True),  # noqa: B008
    top_k: int = Body(default=10),  # noqa: B008
    tier: Optional[int] = Body(default=None),  # noqa: B008
):
    """Run a vector similarity search with no relevance filtering.

    ``tier`` selects which collection to search (1, 2, or 3).
    When ``None``, all tier collections are searched.
    """
    try:
        hits = vector_search(query, top_k=top_k, tier=tier)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"query": query, "top_k": top_k, "results": hits}


# Smart search with relevance filtering
@app.post("/search/relevant")
async def search_relevant(
    query: str = Body(..., embed=True),  # noqa: B008
    top_k: int = Body(default=10),  # noqa: B008
    relevance_threshold: Optional[float] = Body(default=None),  # noqa: B008
    use_adaptive_threshold: bool = Body(default=False),  # noqa: B008
    tier: Optional[int] = Body(default=None),  # noqa: B008
):
    """Run vector search with automatic relevance filtering.

    ``tier`` selects which collection to search (1, 2, or 3).
    When ``None``, all tier collections are searched.
    """
    try:
        return search_with_relevance_filter(
            query=query,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            use_adaptive_threshold=use_adaptive_threshold,
            tier=tier,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Strict relevance search ---
@app.post("/search/strict")
async def search_strict(
    query: str = Body(..., embed=True),  # noqa: B008
    top_k: int = Body(default=10),  # noqa: B008
    tier: Optional[int] = Body(default=None),  # noqa: B008
):
    """Search with strict relevance filtering (threshold=0.7).

    ``tier`` selects which collection to search (1, 2, or 3).
    When ``None``, all tier collections are searched.
    """
    try:
        response = search_with_relevance_filter(
            query=query,
            top_k=top_k,
            relevance_threshold=RelevanceFilter.STRICT_THRESHOLD,
            tier=tier,
        )
        return {
            "query": query,
            "threshold": response["threshold"],
            "results": response["relevant_results"],
            "filtered_count": response["irrelevant_count"],
        }
    except Exception as e:
        logger.error(f"Strict search error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Fact-check endpoint
# ---------------------------------------------------------------------------


class FactCheckRequest(BaseModel):
    """Request body for POST /fact-check."""

    claim: str


class FactCheckResponse(BaseModel):
    """Response body for POST /fact-check."""

    claim: str
    score: float
    reasoning: str
    sources: list[str]


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_endpoint(request: FactCheckRequest):
    """Run the ReAct fact-checker (Gemma 4) against Tier 1 and Tier 2 Milvus collections.

    Accepts a natural-language ``claim``. A LangGraph ReAct agent calls
    ``retrieve_tier1`` and ``retrieve_tier2`` tools, then produces a
    truthfulness score (0-1) with reasoning and source domains.

    **Source tiers and scoring weight**

    | Tier   | Sources                         | Weight in score          |
    |--------|---------------------------------|--------------------------|
    | Tier 1 | Raw / unverified                | Low — indirect only      |
    | Tier 2 | Credible (credibility >= 0.7)   | High — primary evidence  |

    **Score interpretation**

    | Range     | Meaning                                                      |
    |-----------|--------------------------------------------------------------|
    | 0.8 – 1.0 | Strongly supported — tier2 hits with distance >= 0.6        |
    | 0.6 – 0.8 | Moderately supported                                         |
    | 0.4 – 0.6 | Uncertain — mixed signals, tier1 only, or sparse data        |
    | 0.2 – 0.4 | Weak support — few or low-similarity records                 |
    | 0.0 – 0.2 | Contradicted or entirely unsupported                         |
    """
    try:
        result: FactCheckResult = fact_check(request.claim)
    except Exception as e:
        logger.error("fact_check error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return FactCheckResponse(
        claim=request.claim,
        score=result.score,
        reasoning=result.reasoning,
        sources=result.sources,
    )


# ---------------------------------------------------------------------------
# Fact-check test-runner endpoint
# ---------------------------------------------------------------------------

_DATASET_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "claims_dataset.json"


class RunTestsRequest(BaseModel):
    """Request body for POST /fact-check/run-tests."""

    limit: Optional[int] = None


class LabelStats(BaseModel):
    total: int
    passed: int
    avg_score: float


class TestDetail(BaseModel):
    claim: str
    label: str
    score: float
    passed: bool


class RunTestsResponse(BaseModel):
    total: int
    passed: int
    failed: int
    accuracy: float
    by_label: Dict[str, LabelStats]
    details: List[TestDetail]


@app.post("/fact-check/run-tests", response_model=RunTestsResponse)
async def run_tests(request: RunTestsRequest):
    """Run fact-checker against the built-in test dataset and return accuracy metrics.

    Loads ``tests/claims_dataset.json``, runs each case through ``fact_check()``,
    and compares the resulting score against ``expected_score_min``/``expected_score_max``.

    **Request**

    ```json
    { "limit": 100 }
    ```

    ``limit`` caps the number of cases to evaluate (default: all 100).

    **Response fields**

    - ``total`` / ``passed`` / ``failed`` / ``accuracy`` — overall counters
    - ``by_label`` — breakdown by ``true`` / ``partially_true`` / ``fake``
    - ``details`` — per-claim results with score and pass/fail flag
    """
    if not _DATASET_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Dataset not found: {_DATASET_PATH}",
        )

    with _DATASET_PATH.open() as fh:
        dataset: List[Dict[str, Any]] = json.load(fh)

    if request.limit is not None and request.limit > 0:
        dataset = dataset[: request.limit]

    details: List[TestDetail] = []
    label_buckets: Dict[str, Dict[str, Any]] = {}

    for case in dataset:
        claim: str = case["claim"]
        label: str = case["label"]
        score_min: float = float(case["expected_score_min"])
        score_max: float = float(case["expected_score_max"])

        try:
            fc_result: FactCheckResult = fact_check(claim)
            score = fc_result.score
        except Exception as exc:
            logger.error("fact_check failed for claim %r: %s", claim[:60], exc)
            score = 0.0

        passed = score_min <= score <= score_max

        details.append(TestDetail(claim=claim, label=label, score=score, passed=passed))

        bucket = label_buckets.setdefault(label, {"total": 0, "passed": 0, "scores": []})
        bucket["total"] += 1
        bucket["scores"].append(score)
        if passed:
            bucket["passed"] += 1

    total = len(details)
    passed_count = sum(1 for d in details if d.passed)
    failed_count = total - passed_count
    accuracy = passed_count / total if total > 0 else 0.0

    by_label: Dict[str, LabelStats] = {
        lbl: LabelStats(
            total=b["total"],
            passed=b["passed"],
            avg_score=sum(b["scores"]) / len(b["scores"]) if b["scores"] else 0.0,
        )
        for lbl, b in label_buckets.items()
    }

    return RunTestsResponse(
        total=total,
        passed=passed_count,
        failed=failed_count,
        accuracy=accuracy,
        by_label=by_label,
        details=details,
    )


# --- Error handler example ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions and return a JSON error response."""
    return JSONResponse(
        status_code=500, content={"message": f"Unexpected error: {exc}"}
    )

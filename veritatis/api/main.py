"""Main entry point for the veritatis API using FastAPI."""

from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import time, hashlib
import logging

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from veritatis.vector_stores import MilvusRecordStore, ensure_connection, init_collections, ensure_collection_loaded
from pymilvus.orm import utility
from veritatis.embeddings import embedding_generator
from pymilvus import Collection

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
   logger.info("Warming up embedding model (this may take 5-10 seconds on first run)...")
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
   content: str = Body(..., embed=True),
   source_url: Optional[str] = Body(None),
   supabase_id: Optional[str] = Body(None),
   metadata: Optional[Dict[str, Any]] = Body(None)
):
   """Ingest a text record into Tier1 with embeddings and dedup by hash."""
   if _store is None:
      raise HTTPException(status_code=503, detail="Vector store not initialized. Check Milvus connection.")
   
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
      "supabase_id": supabase_id or ""
   }

   try:
      _store.insert_record(_TIER1, record)
   except Exception as e:
      return JSONResponse(status_code=500, content={"error": str(e)})

   return {"id": record_id, "status": "inserted", "collection": _TIER1}

# --- Vector search endpoint ---
@app.post("/search")
async def search(
   query: str = Body(..., embed=True),
   top_k: int = Body(10),
):
   """Run a vector similarity search against Tier1."""
   if _store is None:
      raise HTTPException(status_code=503, detail="Vector store not initialized. Check Milvus connection.")
   
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
         output_fields=["id", "content", "source_url", "credibility_score", "ingested_timestamp", "supabase_id"],
      )
   except Exception as e:
      return JSONResponse(status_code=500, content={"error": str(e)})

   # results is a list per query (we only have one)
   hits = []
   for hit in results[0]:
      hits.append({
         "id": str(hit.id),
         "content": str(getattr(hit, 'content', '')),
         "source_url": str(getattr(hit, 'source_url', '')),
         "credibility_score": float(getattr(hit, 'credibility_score', 0.0)),
         "ingested_timestamp": int(getattr(hit, 'ingested_timestamp', 0)),
         "supabase_id": str(getattr(hit, 'supabase_id', '')),
         "distance": float(hit.distance),
      })

   return {"query": query, "top_k": top_k, "results": hits}

# --- Error handler example ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
   """Handle unexpected exceptions and return a JSON error response."""
   return JSONResponse(
      status_code=500,
      content={"message": f"Unexpected error: {exc}"}
   )

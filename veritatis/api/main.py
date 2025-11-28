from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import time, hashlib

from veritatis.vector_stores import MilvusRecordStore, ensure_connection
from pymilvus.orm import utility
from veritatis.embeddings import embedding_generator
from pymilvus import Collection

_TIER1 = "veritatis_tier1_lake"
_store = MilvusRecordStore()

app = FastAPI(title="Veritatis API", version="1.0")

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
   return {"status": "ok"}

@app.get("/debug/collections")
async def list_collections():
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
   normalized = " ".join(content.split()).strip()
   base = normalized + ("|" + source_url if source_url else "")
   record_id = hashlib.sha256(base.encode("utf-8")).hexdigest()

   # Dedup check
   if _store.record_exists(_TIER1, record_id):
      return {"id": record_id, "status": "duplicate", "collection": _TIER1}

   embedding = embedding_generator.generate(normalized)
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
   vec = embedding_generator.generate(query)
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
      row = hit.entity.get("_raw") if hasattr(hit.entity, "get") else hit.entity
      hits.append({
         "id": row.get("id"),
         "content": row.get("content"),
         "source_url": row.get("source_url"),
         "credibility_score": row.get("credibility_score"),
         "ingested_timestamp": row.get("ingested_timestamp"),
         "supabase_id": row.get("supabase_id"),
         "distance": hit.distance,
      })

   return {"query": query, "top_k": top_k, "results": hits}

# --- Error handler example ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
   return JSONResponse(
      status_code=500,
      content={"message": f"Unexpected error: {exc}"}
   )

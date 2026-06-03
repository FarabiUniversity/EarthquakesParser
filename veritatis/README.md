# Veritatis: Earthquake News RAG

This repo provides a FastAPI service and Milvus-backed vector store to ingest, and validate earthquake-related content via a tiered pipeline.

## Quick Start

- Requirements: macOS/ARM64, Docker, Python 3.12, Poetry (or venv + pip).
- Services: etcd + Milvus standalone + FastAPI API.

### Environment

- Configure `.env` (for local dev) or real environment variables:
  - `ENV=development`
  - `MILVUS_HOST=localhost`
  - `MILVUS_PORT=19530`

The code prefers real env vars; `.env` is only used when `ENV` is missing or `ENV=development`.

### Run Services

- Start Milvus + etcd + API:

```zsh
docker compose up -d --build
```

- If port `8000` is busy, either stop the other service or remap in `docker-compose.yml` (e.g., `8080:8000`).

### API Endpoints

- `GET /health` — health check
- `POST /ingest` — body `{ text: str, iid: str, tier?: int, credibility_score?: float, date?: int, domain?: str }` → global-dedup, embed, insert into a tier
- `POST /move` — body `{ iids: list[str], source_tier: int, target_tier: int }` → move records between tiers
- `POST /consensus/analyze` — analyze Tier 1 using Milvus vectors + Supabase `main_text`, optionally promote best to Tier 2

Run API locally:

```zsh
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Move a record between tiers (example: tier1 → tier2):

```zsh
curl -sS -X POST "http://127.0.0.1:8000/move" \
  -H "Content-Type: application/json" \
  -d '{"iids":["181abbb9-8211-45bc-9c84-51add70667e2"],"source_tier":1,"target_tier":2}'
```

### Milvus Collections

Three collections with `FLOAT_VECTOR(384)` embeddings (default SentenceTransformers model):
- `veritatis_tier1_lake`: raw lake
- `veritatis_tier2_arena`: candidate facts
- `veritatis_tier3_sanctum`: verified facts

HNSW index with COSINE similarity is created on `embedding`.

Initialize (if not created yet):

```zsh
poetry run python scripts/init_collections.py
```

### Embeddings

- Default model: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, normalized).
- Override with `EMBED_MODEL_NAME`.
- If you change the model to one with a different dimension, you must also update the Milvus collection schema (see `veritatis/veritatis/vector_stores.py:init_collections`) and recreate/wipe Milvus state.

### Testing

Run all tests (integration tests auto-skip if Milvus isn't reachable):

```zsh
poetry run pytest -q
```

Unit tests only:

```zsh
poetry run pytest -q tests/test_vector_consensus.py tests/test_embeddings.py
```

Embeddings unit test (downloads the embedding model on first run):

```zsh
poetry run pytest -q tests/test_embeddings.py
```

Vector store integration tests (real Milvus required):

```zsh
# Start Milvus services first
docker compose up -d etcd milvus

# Run integration tests against real Milvus on localhost:19530
poetry run pytest -q tests/test_vector_store_integration.py
```

Optional artifact for embeddings (JUnit XML):

```zsh
poetry run pytest -q tests/test_embeddings.py --junitxml=artifacts/embeddings_junit.xml
```

### Milvus Connectivity

If you want integration tests (real Milvus):
- Ensure `milvus-standalone` is Running and `localhost:19530` is reachable.
- Common fixes on macOS/ARM64:
  - Use a recent image (e.g., `milvusdb/milvus:v2.4.x`) if `v2.3.0` exits.
  - Verify logs: `docker logs milvus-standalone`.
  - Check ports: `nc -zv 127.0.0.1 19530`.

### Development Notes

- `veritatis/vector_stores.py` exposes `MilvusRecordStore` with:
  - `insert_record(collection, record|list[record])`
  - `record_exists(collection, iid)`
  - `find_record_collection(iid)` (search across tier collections)
  - `get_record(collection, iid)`
  - `move_records(src, dst, iids)`
- Set `MILVUS_SKIP_CONNECT=1` to avoid real connections in tests.
- `veritatis/embeddings.py` provides normalized embeddings via `EmbeddingGenerator`.

### Dedup semantics

- `iid` is the primary key and is expected to equal `parsed_content.id` in Supabase.
- Ingestion performs *global* dedup across all tier collections: the same `iid` cannot exist in two tiers.

### Troubleshooting

- Model download interrupted: rerun, or switch to a lighter model for dev.
- Port conflicts: free port `8000` or change compose mapping.
- Milvus not reachable: confirm the container is Up; consider upgrading the image.

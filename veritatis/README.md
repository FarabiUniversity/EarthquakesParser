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
- `POST /ingest` — body `{ content: str, source_url?: str }` → dedup, embed, insert into Tier 1

Run API locally:

```zsh
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Milvus Collections

Three collections with `FLOAT_VECTOR(1024)` embeddings (bge-m3):
- `veritatis_tier1_lake`: raw lake
- `veritatis_tier2_arena`: candidate facts
- `veritatis_tier3_sanctum`: verified facts

HNSW index with COSINE similarity is created on `embedding`.

Initialize (if not created yet):

```zsh
poetry run python scripts/init_collections.py
```

### Embeddings

- Default model: `BAAI/bge-m3` (1024 dims, normalized).
- First run downloads the model (~GBs). Allow time or pre-bake/caching.

### Testing

- Embeddings unit test:

```zsh
poetry run pytest -q tests/test_embeddings.py
```

- Vector store integration tests (real Milvus required):

```zsh
# Start Milvus services first
docker compose up -d etcd milvus

# Run integration tests against real Milvus on localhost:19530
poetry run pytest -q tests/test_vector_store_unit.py
```

- Optional artifact for embeddings (JUnit XML):

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
  - `record_exists(collection, id)`
  - `get_record(collection, id)` (uses client.get to include vectors)
  - `move_record(src, dst, id)`
- Set `MILVUS_SKIP_CONNECT=1` to avoid real connections in tests.
- `veritatis/embeddings.py` provides normalized embeddings via `EmbeddingGenerator`.

### Troubleshooting

- Model download interrupted: rerun, or switch to a lighter model for dev.
- Port conflicts: free port `8000` or change compose mapping.
- Milvus not reachable: confirm the container is Up; consider upgrading the image.

### Next Steps

- Supabase integration for ID mapping.
- Reranking for search results.
- Additional CRUD endpoints for tier movement.

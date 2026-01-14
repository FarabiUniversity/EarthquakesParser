# Veritatis

Veritatis is a small FastAPI + Milvus service for embedding and searching earthquake-related text.

## Data model (Tier 1)

All ingestion uses a simple, stable, “tier-1” shape:

```json
{
  "id": "...",
  "text": "...",
  "metadata": {"any": "json"}
}
```

- `id` is your primary key (string).
- `text` is the content that gets embedded.
- `metadata` is any JSON you want to carry along (stored as JSON-serialized text in Milvus).

## Requirements

- Docker (for Milvus)
- Python 3.11+ (repo targets 3.11–3.12)
- `uv` (recommended) or any equivalent environment manager

## Setup

Install deps:

```bash
uv sync
```

Create `.env` (example):

```dotenv
MILVUS_HOST=localhost
MILVUS_PORT=19530

SUPABASE_URL=...
SUPABASE_KEY=...
```

Notes:

- `SUPABASE_KEY` can be an anon key only if your table policies allow read access.
- For backend ingestion, prefer `SUPABASE_SERVICE_ROLE_KEY` (the script accepts either).

## Start Milvus (Docker)

```bash
docker-compose up -d etcd minio milvus
```

Confirm Milvus is healthy:

```bash
curl -sf http://localhost:9091/healthz
```

## Run the API (local)

Run FastAPI on port 8001:

```bash
set -a && source .env && set +a
ENV=development MILVUS_RECREATE_ON_STARTUP=true \
  uv run uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Use `MILVUS_RECREATE_ON_STARTUP=true` when:

- you changed the Milvus schema (fields/max_length/dim), or
- you want a clean slate.

For normal restarts, set it to `false`.

## Ingest parsed content from Supabase

The script in veritatis/ingest_parsed_content.py reads `parsed_content` from Supabase and POSTs batches to the API.

In a second terminal:

```bash
set -a && source .env && set +a
uv run python veritatis/ingest_parsed_content.py
```

Notes:

- The script is resilient to missing columns like `parsed_at`.
- If a `main_text` row contains JSON like `[]` or `["..."]`, it is normalized into plain text before embedding.
- Long documents are chunked into multiple records (`<parent_id>:<chunk_index>`) to fit Milvus VARCHAR limits.

## API examples

Health:

```bash
curl -s http://127.0.0.1:8001/health
```

Ingest (batch):

```bash
curl -s http://127.0.0.1:8001/ingest \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "example-1",
      "text": "Magnitude 5.2 earthquake near City X.",
      "metadata": {"source": "demo"}
    }
  ]'
```

Search:

```bash
curl -s http://127.0.0.1:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query":"earthquake","top_k":5}'
```

Move Tier1 → Tier2:

```bash
curl -s http://127.0.0.1:8001/move \
  -H "Content-Type: application/json" \
  -d '{"record_id":"example-1","verification_confidence":0.8,"cross_source_count":2}'
```

## Testing

```bash
uv run pytest -q
```

## Troubleshooting

- **Schema mismatch errors**: set `MILVUS_RECREATE_ON_STARTUP=true` (or wipe Milvus volumes) and restart.
- **Supabase permission errors**: your key likely lacks read access to `parsed_content` under RLS; use service role or adjust policies.
- **Port conflicts**: change the `--port` argument when running uvicorn locally.

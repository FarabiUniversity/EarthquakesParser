# KNDC Parser Guide

This guide explains how the KNDC ingestion flow works now that parser output is written to Supabase instead of local JSON snapshot files.

## What the system does

The KNDC codebase has two ingestion paths:

- `KndcParser` fetches individual KNDC news items by `newsid`.
- `KndcBulletinHourlyParser` and `KndcBulletinPopulator` fetch KNDC bulletin rows by `event_id`.

Both paths normalize the KNDC payload, then upsert the result into Supabase using a shared helper:

- [earthquakes_parser/parser/kndc_supabase.py](../earthquakes_parser/parser/kndc_supabase.py)

The scheduler combines both flows so it can keep collecting news and bulletins continuously.

## Data flow

The typical flow is:

1. Fetch raw KNDC JSON from the remote KNDC endpoints.
2. Normalize the payload into a stable Python dictionary.
3. Convert the record into Supabase-safe values.
4. Upsert the row into Supabase.
5. Retry transient HTTP/Supabase transport failures automatically.

For news items, the natural key is `newsid`. For bulletins, the natural key is `event_id`.

## Supabase tables

Create these two tables in Supabase:

### `kndc_news`

Stores individual KNDC news items.

Important columns:

- `id` UUID primary key
- `newsid` BIGINT unique
- `source_url` TEXT unique
- `title` TEXT
- `published_at` TEXT
- `main_text` TEXT
- `raw` JSONB
- `created_at`, `updated_at` timestamps

### `kndc_bulletins`

Stores KNDC bulletin rows.

Important columns:

- `id` UUID primary key
- `event_id` BIGINT unique
- `source_url` TEXT unique
- `epoch_time` BIGINT
- `occurred_at_utc` TIMESTAMPTZ
- `latitude`, `longitude`, `depth_km`
- `mb`, `mpv`, `energy_class_k`
- `geographic_region`, `seismic_region`, `quality`, `author`
- `last_updated` TEXT
- `parsed` JSONB
- `enriched_text` TEXT
- `raw` JSONB
- `created_at`, `updated_at` timestamps

The migration file is here:

- [supabase/migrations/20260617_create_kndc_ingestion_schema.sql](../supabase/migrations/20260617_create_kndc_ingestion_schema.sql)

## How deduplication works

The system does not rely on local files anymore.

### News

`KndcSupabaseStore.latest_newsid()` queries Supabase for the highest stored `newsid`. The scheduler uses that value as the next cursor.

### Bulletins

`KndcSupabaseStore.bulletin_exists(event_id)` checks whether that exact bulletin row already exists. If it does, the row is skipped; otherwise it is upserted.

This means re-running the scheduler is safe.

## Retry behavior

Supabase writes and reads are wrapped with a small retry budget in `KndcSupabaseStore`.

That protects the pipeline from temporary network issues such as:

- HTTP/2 connection termination
- `httpx.RemoteProtocolError`
- transient Supabase transport glitches

If a request still fails after the retry budget, the exception is raised and logged.

## How to run it

Make sure these environment variables are available:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_KEY`

The repository already loads them from the repo-local `.env` file and from `veritatis/.env`.

### Run the scheduler

```bash
python -m earthquakes_parser.parser.kndc_scheduler --interval 3600
```

This runs both flows:

- bulletin ingestion first
- news probing next

### Run a manual news batch

```bash
python -m earthquakes_parser.parser.kndc_parser --newsids 5,6,7
```

### Run bulletin backfill

```bash
python -m earthquakes_parser.parser.kndc_bulletin_populate
```

## Practical notes

- The old JSON snapshot files are no longer the source of truth.
- Supabase is now the source of truth for KNDC records.
- The scheduler is for incremental collection, not deep historical recovery.
- For a historical recovery of news ids, run `KndcParser` with the ids you want.

## Code entry points

- [earthquakes_parser/parser/kndc_parser.py](../earthquakes_parser/parser/kndc_parser.py)
- [earthquakes_parser/parser/kndc_bulletin_parser.py](../earthquakes_parser/parser/kndc_bulletin_parser.py)
- [earthquakes_parser/parser/kndc_bulletin_populate.py](../earthquakes_parser/parser/kndc_bulletin_populate.py)
- [earthquakes_parser/parser/kndc_scheduler.py](../earthquakes_parser/parser/kndc_scheduler.py)
- [earthquakes_parser/parser/kndc_supabase.py](../earthquakes_parser/parser/kndc_supabase.py)

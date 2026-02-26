"""Re-ingest already-ingested Supabase rows into Milvus via the Veritatis API.

Use this when Milvus collections were dropped/wiped but Supabase rows have already
been marked with `status='ingested'`.

This script:
- fetches rows from Supabase table `parsed_content` where `status='ingested'`
- normalizes + chunks `main_text`
- POSTs each chunk to the Veritatis API `POST /ingest`
- does NOT update Supabase statuses (to avoid infinite loops / unwanted changes)

Run examples:
- From host (with deps installed):
    python3 -m veritatis.reingest_ingested_content

- From docker (preferred if you use docker-compose):
    docker compose exec api python -m veritatis.reingest_ingested_content

Environment:
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)
- INGEST_URL (default: http://localhost:8000/ingest)

Optional args:
- --batch-size 200
- --sleep 0.1
- --max-records 1000
- --start-offset 0
"""

from __future__ import annotations

import argparse
import os
import time

from veritatis.ingest_parsed_content import BATCH_SIZE as DEFAULT_BATCH_SIZE
from veritatis.ingest_parsed_content import SLEEP_BETWEEN_BATCHES as DEFAULT_SLEEP
from veritatis.ingest_parsed_content import fetch_parsed_content, send_to_ingest


def parse_args() -> argparse.Namespace:
    """Produce args needed for reingestion."""
    parser = argparse.ArgumentParser(description="Re-ingest Supabase ingested rows")
    parser.add_argument(
        "--status",
        default=os.getenv("REINGEST_SOURCE_STATUS", "ingested"),
        help="Source status to pull from Supabase (default: ingested)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.getenv("SLEEP_BETWEEN_BATCHES", str(DEFAULT_SLEEP))),
        help="Seconds to sleep between batches (default from ingest_parsed_content)",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=int(os.getenv("START_OFFSET", "0")),
        help="Starting offset for paging (default: 0)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=int(os.getenv("MAX_RECORDS", "0")),
        help="Maximum number of Supabase rows to process (0 = no limit)",
    )
    return parser.parse_args()


def main() -> None:
    """Docstring for main."""
    args = parse_args()

    status = str(args.status)
    batch_size = int(args.batch_size)
    sleep_s = float(args.sleep)
    offset = int(args.start_offset)
    max_records = int(args.max_records)

    processed = 0

    while True:
        if max_records and processed >= max_records:
            print(f"✅ Reached max_records={max_records}; stopping.")
            break

        limit = batch_size
        if max_records:
            limit = min(limit, max_records - processed)

        batch = fetch_parsed_content(offset, limit, status=status)
        if not batch:
            print("✅ No more records to re-ingest.")
            break

        print(
            f"➡️ Re-ingesting batch offset={offset} size={len(batch)} status={status}"
        )
        send_to_ingest(batch)

        processed += len(batch)
        offset += len(batch)

        if sleep_s:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()

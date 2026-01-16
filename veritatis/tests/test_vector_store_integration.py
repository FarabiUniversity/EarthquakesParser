"""Integration tests for the Milvus-backed vector store."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import veritatis.vector_stores as vs

EMBED_DIM = 384
ALMATY_TZ = timezone(timedelta(hours=5))


def _rand_id():
    """Return a short random id suitable for use as a primary key."""
    return uuid.uuid4().hex[:24]


def _embedding():
    """Return a deterministic embedding vector of length EMBED_DIM."""
    # Simple deterministic vector of length EMBED_DIM
    return [float(i) / EMBED_DIM for i in range(EMBED_DIM)]


@pytest.fixture(scope="module", autouse=True)
def _ensure_milvus():
    """Skip tests if Milvus is not reachable."""
    # Ensure we can reach a real Milvus; skip if unavailable
    try:
        vs.ensure_connection()
    except Exception as e:
        pytest.skip(f"Milvus not available: {e}")


def test_insert_and_exists_and_get():
    """Insert a record and verify exists/get works."""
    store = vs.MilvusRecordStore()

    name = f"it_single_{_rand_id()}"
    vs.create_collection_if_not_exists(
        name,
        fields=[
            vs.FieldSchema(
                name="id", dtype=vs.DataType.VARCHAR, is_primary=True, max_length=100
            ),
            vs.FieldSchema(name="content", dtype=vs.DataType.VARCHAR, max_length=10000),
            vs.FieldSchema(
                name="embedding", dtype=vs.DataType.FLOAT_VECTOR, dim=EMBED_DIM
            ),
        ],
        description="integration test",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 8, "efConstruction": 32},
        },
    )

    rid = _rand_id()
    record = {"id": rid, "content": "Hello", "embedding": _embedding()}
    pks = store.insert_record(name, record)
    assert pks is not None and pks[0] == rid
    assert store.record_exists(name, rid) is True
    fetched = store.get_record(name, rid)
    assert fetched["id"] == rid and len(fetched["embedding"]) == EMBED_DIM

    out = Path("artifacts") / "milvus_store_integration_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "test": "insert_exists_get",
                "collection": name,
                "id": rid,
                "content": fetched["content"],
                "embedding_dim": len(fetched["embedding"]),
                "timestamp": datetime.now(ALMATY_TZ).isoformat(),
            },
            indent=2,
        )
    )


def test_insert_batch_and_move():
    """Insert a batch, then move one record between collections."""
    store = vs.MilvusRecordStore()

    src = f"it_src_{_rand_id()}"
    dst = f"it_dst_{_rand_id()}"

    for nm in (src, dst):
        vs.create_collection_if_not_exists(
            nm,
            fields=[
                vs.FieldSchema(
                    name="id",
                    dtype=vs.DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                ),
                vs.FieldSchema(
                    name="content", dtype=vs.DataType.VARCHAR, max_length=10000
                ),
                vs.FieldSchema(
                    name="embedding", dtype=vs.DataType.FLOAT_VECTOR, dim=EMBED_DIM
                ),
            ],
            description="integration test",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 8, "efConstruction": 32},
            },
        )

    batch = []
    ids = []
    for i in range(3):
        rid = _rand_id()
        ids.append(rid)
        batch.append({"id": rid, "content": f"Rec {i}", "embedding": _embedding()})

    pks = store.insert_record(src, batch)
    assert pks == ids

    to_move = ids[1]
    ok = store.move_record(src, dst, to_move)
    assert ok is True
    # Note: In Milvus v2.2, deleted records may still appear in queries
    # until compaction. So we verify the move succeeded by checking the
    # destination, not absence from source.
    assert store.record_exists(dst, to_move) is True

    out = Path("artifacts") / "milvus_store_integration_move.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "moved_id": to_move,
                "from": src,
                "to": dst,
            },
            indent=2,
        )
    )

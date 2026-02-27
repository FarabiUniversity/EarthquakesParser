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
    try:
        vs.ensure_connection()
    except Exception as e:
        pytest.skip(f"Milvus not available: {e}")


def _create_test_collection(name):
    """Create a test collection with the veritatis schema (no tier field)."""
    vs.create_collection_if_not_exists(
        name,
        fields=[
            vs.FieldSchema(
                name="iid", dtype=vs.DataType.VARCHAR, is_primary=True, max_length=64
            ),
            vs.FieldSchema(
                name="embedding", dtype=vs.DataType.FLOAT_VECTOR, dim=EMBED_DIM
            ),
            vs.FieldSchema(name="credibility_score", dtype=vs.DataType.FLOAT),
            vs.FieldSchema(name="date", dtype=vs.DataType.INT64),
            vs.FieldSchema(name="domain", dtype=vs.DataType.VARCHAR, max_length=500),
        ],
        description="integration test",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 32},
        },
    )


def test_insert_and_exists_and_get():
    """Insert a record and verify exists/get works."""
    store = vs.MilvusRecordStore()

    name = f"it_single_{_rand_id()}"
    _create_test_collection(name)

    rid = _rand_id()
    record = {
        "iid": rid,
        "embedding": _embedding(),
        "credibility_score": 0.0,
        "date": 0,
        "domain": "example.com",
    }
    pks = store.insert_record(name, record)
    assert pks is not None and pks[0] == rid
    assert store.record_exists(name, rid) is True
    fetched = store.get_record(name, rid)
    assert fetched["iid"] == rid and len(fetched["embedding"]) == EMBED_DIM

    out = Path("artifacts") / "milvus_store_integration_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "test": "insert_exists_get",
                "collection": name,
                "iid": rid,
                "embedding_dim": len(fetched["embedding"]),
                "timestamp": datetime.now(ALMATY_TZ).isoformat(),
            },
            indent=2,
        )
    )


def test_insert_batch_and_move():
    """Insert a batch into one collection, then move a record to another."""
    store = vs.MilvusRecordStore()

    src_name = f"it_src_{_rand_id()}"
    tgt_name = f"it_tgt_{_rand_id()}"
    _create_test_collection(src_name)
    _create_test_collection(tgt_name)

    batch = []
    ids = []
    for i in range(3):
        rid = _rand_id()
        ids.append(rid)
        batch.append(
            {
                "iid": rid,
                "embedding": _embedding(),
                "credibility_score": 0.5 + i * 0.1,
                "date": 0,
                "domain": f"source{i}.com",
            }
        )

    pks = store.insert_record(src_name, batch)
    assert pks == ids

    # Move second record from src to tgt
    count = store.move_records(src_name, tgt_name, [ids[1]])
    assert count == 1

    # Verify it's gone from source and present in target
    assert store.record_exists(src_name, ids[1]) is False
    fetched = store.get_record(tgt_name, ids[1])
    assert fetched is not None and fetched["iid"] == ids[1]

    out = Path("artifacts") / "milvus_store_integration_tier.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "moved_iid": ids[1],
                "source": src_name,
                "target": tgt_name,
            },
            indent=2,
        )
    )

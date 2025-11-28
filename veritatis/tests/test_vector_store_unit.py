import os
import json
from types import SimpleNamespace
from datetime import datetime

# Ensure vector_stores doesn't try to connect at import
os.environ["MILVUS_SKIP_CONNECT"] = "1"

import uuid
import pytest

# Import module after env var set
import veritatis.vector_stores as vs

EMBED_DIM = 1024

class FakeField:
	def __init__(self, name):
		self.name = name

class FakeSchema:
	def __init__(self, fields):
		self.fields = [FakeField(f) for f in fields]

class FakeCollection:
	_db = {}

	def __init__(self, name, schema=None):
		self.name = name
		if name not in FakeCollection._db:
			FakeCollection._db[name] = {"rows": {}, "schema": schema}
		self.schema = FakeSchema(["id", "content", "embedding"]) if schema is None else schema

	def insert(self, data_columns):
		ids, contents, embeddings = data_columns
		for i, rid in enumerate(ids):
			FakeCollection._db[self.name]["rows"][rid] = {
					"id": rid,
					"content": contents[i],
					"embedding": embeddings[i],
			}
		return SimpleNamespace(primary_keys=list(ids))

	def delete(self, expr: str):
		# expr format: id == '...'
		rid = expr.split("==")[-1].strip().strip("'").strip('"')
		FakeCollection._db[self.name]["rows"].pop(rid, None)

	def query(self, expr: str, output_fields=None):
		rid = expr.split("==")[-1].strip().strip("'").strip('"')
		row = FakeCollection._db[self.name]["rows"].get(rid)
		return [row] if row else []

	def create_index(self, field_name: str, index_params: dict):
		# No-op for unit test
		return True

# Monkeypatch targets in module namespace
@pytest.fixture(autouse=True)
def _patch_collection(monkeypatch):
	monkeypatch.setattr(vs, "Collection", FakeCollection)
	monkeypatch.setattr(vs, "utility", SimpleNamespace(has_collection=lambda name: name in FakeCollection._db))
	yield
	# teardown
	FakeCollection._db.clear()


def _rand_id():
	return uuid.uuid4().hex[:24]


def _embedding():
	return [float(i)/EMBED_DIM for i in range(EMBED_DIM)]


def test_insert_and_exists_and_get(tmp_path):
	class FakeClient:
		@staticmethod
		def get(name, ids):
			rows = []
			table = FakeCollection._db[name]["rows"]
			for rid in ids:
					if rid in table:
						rows.append(table[rid])
			return rows
	store = vs.MilvusRecordStore(uri="http://fake:19530", client=FakeClient())

	name = f"unit_single_{_rand_id()}"
	vs.create_collection_if_not_exists(name, fields=[
		vs.FieldSchema(name="id", dtype=vs.DataType.VARCHAR, is_primary=True, max_length=100),
		vs.FieldSchema(name="content", dtype=vs.DataType.VARCHAR, max_length=10000),
		vs.FieldSchema(name="embedding", dtype=vs.DataType.FLOAT_VECTOR, dim=EMBED_DIM),
	], description="unit test", index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 32}})

	rid = _rand_id()
	record = {"id": rid, "content": "Hello", "embedding": _embedding()}
	pks = store.insert_record(name, record)
	assert pks is not None and pks[0] == rid
	assert store.record_exists(name, rid) is True
	fetched = store.get_record(name, rid)
	assert fetched["id"] == rid and len(fetched["embedding"]) == EMBED_DIM

	# Save a small results artifact
	out = tmp_path / "milvus_store_unit_results.json"
	out.write_text(json.dumps({
		"test": "insert_exists_get",
		"collection": name,
		"id": rid,
		"content": fetched["content"],
		"embedding_dim": len(fetched["embedding"]),
		"timestamp": datetime.utcnow().isoformat() + "Z",
	}, indent=2))


def test_insert_batch_and_move(tmp_path):
	class FakeClient:
		@staticmethod
		def get(name, ids):
			rows = []
			table = FakeCollection._db[name]["rows"]
			for rid in ids:
					if rid in table:
						rows.append(table[rid])
			return rows
	store = vs.MilvusRecordStore(uri="http://fake:19530", client=FakeClient())

	src = f"unit_src_{_rand_id()}"
	dst = f"unit_dst_{_rand_id()}"

	for nm in (src, dst):
		vs.create_collection_if_not_exists(nm, fields=[
			vs.FieldSchema(name="id", dtype=vs.DataType.VARCHAR, is_primary=True, max_length=100),
			vs.FieldSchema(name="content", dtype=vs.DataType.VARCHAR, max_length=10000),
			vs.FieldSchema(name="embedding", dtype=vs.DataType.FLOAT_VECTOR, dim=EMBED_DIM),
		], description="unit test", index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 32}})

	batch = []
	ids = []
	for i in range(3):
		rid = _rand_id()
		ids.append(rid)
		batch.append({"id": rid, "content": f"Rec {i}", "embedding": _embedding()})

	pks = store.insert_record(src, batch)
	assert pks == ids

	# Move the middle record
	to_move = ids[1]
	ok = store.move_record(src, dst, to_move)
	assert ok is True
	assert store.record_exists(src, to_move) is False
	assert store.record_exists(dst, to_move) is True

	# Save artifact
	out = tmp_path / "milvus_store_unit_move.json"
	out.write_text(json.dumps({
		"moved_id": to_move,
		"from": src,
		"to": dst,
		"remaining_in_src": [k for k in FakeCollection._db[src]["rows"].keys()],
		"present_in_dst": [k for k in FakeCollection._db[dst]["rows"].keys()],
	}, indent=2))

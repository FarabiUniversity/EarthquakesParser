import os
import json
from types import SimpleNamespace
from datetime import datetime, UTC
import uuid

# Avoid real Milvus connections
os.environ["MILVUS_SKIP_CONNECT"] = "1"

import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)
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
		class _Res:
			primary_keys = list(ids)
		return _Res()

	def delete(self, expr: str):
		rid = expr.split("==")[-1].strip().strip("'").strip('"')
		FakeCollection._db[self.name]["rows"].pop(rid, None)

	def query(self, expr: str, output_fields=None):
		rid = expr.split("==")[-1].strip().strip("'").strip('"')
		row = FakeCollection._db[self.name]["rows"].get(rid)
		return [row] if row else []

	def create_index(self, field_name: str, index_params: dict):
		return True

# Patch module symbols
vs.Collection = FakeCollection
vs.utility = SimpleNamespace(has_collection=lambda n: n in FakeCollection._db)

class FakeClient:
	@staticmethod
	def get(name, ids):
		rows = []
		table = FakeCollection._db[name]["rows"]
		for rid in ids:
			if rid in table:
					rows.append(table[rid])
		return rows

def _rand_id():
	return uuid.uuid4().hex[:24]


def _embedding():
	return [float(i)/EMBED_DIM for i in range(EMBED_DIM)]


def main():
	store = vs.MilvusRecordStore(uri="http://fake:19530", client=FakeClient())
	# Create two collections
	src = f"offline_src_{_rand_id()}"
	dst = f"offline_dst_{_rand_id()}"
	for nm in (src, dst):
		vs.create_collection_if_not_exists(nm, fields=[
			vs.FieldSchema(name="id", dtype=vs.DataType.VARCHAR, is_primary=True, max_length=100),
			vs.FieldSchema(name="content", dtype=vs.DataType.VARCHAR, max_length=10000),
			vs.FieldSchema(name="embedding", dtype=vs.DataType.FLOAT_VECTOR, dim=EMBED_DIM),
		], description="offline unit run", index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 32}})

	# Single insert
	rid_single = _rand_id()
	store.insert_record(src, {"id": rid_single, "content": "Hello offline", "embedding": _embedding()})

	# Batch insert
	ids = []
	batch = []
	for i in range(3):
		rid = _rand_id()
		ids.append(rid)
		batch.append({"id": rid, "content": f"Offline {i}", "embedding": _embedding()})
	store.insert_record(src, batch)

	# Move one
	moved_id = ids[1]
	store.move_record(src, dst, moved_id)

	# Prepare results
	results = {
		"timestamp": datetime.now(UTC).isoformat(),
		"collections": list(FakeCollection._db.keys()),
		"src_count": len(FakeCollection._db[src]["rows"]),
		"dst_count": len(FakeCollection._db[dst]["rows"]),
		"sample_src_keys": list(FakeCollection._db[src]["rows"].keys()),
		"sample_dst_keys": list(FakeCollection._db[dst]["rows"].keys()),
		"moved_id": moved_id,
		"fetched_single": store.get_record(src, rid_single) is not None,
	}

	out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "milvus_store_offline_results.json")
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(results, f, indent=2)
	print(out_path)

if __name__ == "__main__":
	main()

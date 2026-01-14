Testing ingestion : """
curl -s -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Magnitude 5.2 earthquake reported near Almaty, Kazakhstan. Preliminary depth 10 km.",
    "source_url": "https://example.com/news/almaty-quake-1",
    "supabase_id": ""
  }'

"""

out (first time): """
{"id":"<sha256_id>","status":"inserted","collection":"veritatis_tier1_lake"}
"""

out (re-ingest same content/source_url): """
HTTP/1.1 409 Conflict
{"detail":"Duplicate record id: <sha256_id>"}
"""

Now let's search for it: """
curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d ' {"query":"Almaty Earthquake","top_k":3}'
"""

out:"""
{"query":"Almaty Earthquake","top_k":3,"results":[{"id":"<sha256_id>","content":"Magnitude 5.2 earthquake reported near Almaty, Kazakhstan. Preliminary depth 10 km.","source_url":"https://example.com/news/almaty-quake-1","credibility_score":0.0,"ingested_timestamp":1767742397288,"supabase_id":"","distance":0.7740042209625244}]}%
"""

"""
yelnur@MacBook-Air-Yelnur veritatis % curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d ' {"query":"Drunken Sailor","top_k":3}'

>>>>>>
{"query":"Drunken Sailor","top_k":3,"results":[{"id":"<sha256_id>","content":"Magnitude 5.2 earthquake reported near Almaty, Kazakhstan. Preliminary depth 10 km.","source_url":"https://example.com/news/almaty-quake-1","credibility_score":0.0,"ingested_timestamp":1767742397288,"supabase_id":"","distance":-0.07328087091445923}]}%
"""
test embeddings using: "poetry run pytest -q tests/test_embeddings.py"

test vector store using: "poetry run pytest -q tests/test_vector_store_integration.py"

---

Move endpoint (/move) example

1) Ingest a Tier 1 record and capture its id:

"""
ingest_json=$(curl -s -X POST http://localhost:8000/ingest \
	-H 'Content-Type: application/json' \
	-d '{
		"content": "Test event: Magnitude 5.2 earthquake near Almaty on 2026-01-14.",
		"source_url": "https://example.com/test"
	}')

echo "$ingest_json"

record_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$ingest_json")
echo "record_id=$record_id"
"""

2) Move it from Tier 1 -> Tier 2:

"""
curl -s -X POST http://localhost:8000/move \
	-H 'Content-Type: application/json' \
	-d "{\"record_id\":\"$record_id\",\"verification_confidence\":0.42,\"cross_source_count\":2}"
"""

expected out:
"""
{"id":"<same_id>","status":"moved","from":"veritatis_tier1_lake","to":"veritatis_tier2_arena"}
"""

3) Sanity checks:

- Moving again should return 404 (no longer in Tier 1)
- Re-ingesting the same content/source_url should return 409 (exists in Tier 2)

The test_vector_store_integration.py explanation:

	test_insert_and_exists_and_get:

		Creates a temporary collection with VARCHAR id, content, and 384-dim FLOAT_VECTOR fields
		Inserts a single record with deterministic embedding
		Verifies the record was inserted (checks primary key returned)
		Checks record_exists() returns True
		Retrieves the record with get_record() and validates id and embedding dimensions match
		Writes test results to JSON file with Almaty timezone timestamp


	test_insert_batch_and_move:

		Creates two temporary collections (source and destination)
		Inserts a batch of 3 records into source collection
		Moves one record from source to destination using move_record()
		Verifies the moved record exists in the destination collection
		Writes move operation details to JSON file

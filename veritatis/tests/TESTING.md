Testing ingestion : """
curl -X POST "http://localhost:8000/ingest" -H "Content-Type: application/json" -d '{"id":"test_123",
"content":"This is a test document about artificial intelligence and machine learning.","tier":"tier1"}'
"""

out: """
{"id":"965c86292ac7d64a23e614f4c782ad7cc24d9a0eb2063a393a70e70a9e2616ad",
"status":"inserted","collection":"veritatis_tier1_lake"}
"""

Now let's search for it: """
curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d ' {"query":"artificial intelligence","tier":"tier1","top_k":3}'
"""

out:"""
{"query":"artificial intelligence","top_k":3,"results":[{"id":"965c86292ac7d64a23e614f4c782ad7cc24d9a0eb2063a393a70e70a9e2616ad",
"content":"This is a test document about artificial intelligence and machine learning.","source_url":"",
"credibility_score":0.0,"ingested_timestamp":1764709422542,"supabase_id":"",
"distance":0.5481607913970947}]}
"""


test embeddings using: "poetry run pytest -q tests/test_embeddings.py"

test vector store using: "poetry run pytest -q tests/test_vector_store_integration.py"

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
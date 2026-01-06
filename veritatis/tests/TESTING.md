Testing ingestion : """
curl -s -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Magnitude 5.2 earthquake reported near Almaty, Kazakhstan. Preliminary depth 10 km.",
    "source_url": "https://example.com/news/almaty-quake-1",
    "supabase_id": ""
  }'

"""

out: """
{"id":"f953da113cb3cebb8597ed8d3f8d8125ee6367c3f97bd8b93d084fd830ccb775","status":"duplicate","collection":"veritatis_tier1_lake"}
"""

Now let's search for it: """
curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d ' {"query":"Almaty Earthquake","top_k":3}'
"""

out:"""
{"query":"Almaty Earthquake","top_k":3,"results":[{"id":"f953da113cb3cebb8597ed8d3f8d8125ee6367c3f97bd8b93d084fd830ccb775","content":"Magnitude 5.2 earthquake reported near Almaty, Kazakhstan. Preliminary depth 10 km.","source_url":"https://example.com/news/almaty-quake-1","credibility_score":0.0,"ingested_timestamp":1767742397288,"supabase_id":"","distance":0.7740042209625244}]}%  
"""

"""
yelnur@MacBook-Air-Yelnur veritatis % curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d ' {"query":"Drunken Sailor","top_k":3}'

>>>>>>

{"query":"Drunken Sailor","top_k":3,"results":[{"id":"f953da113cb3cebb8597ed8d3f8d8125ee6367c3f97bd8b93d084fd830ccb775","content":"Magnitude 5.2 earthquake reported near Almaty, Kazakhstan. Preliminary depth 10 km.","source_url":"https://example.com/news/almaty-quake-1","credibility_score":0.0,"ingested_timestamp":1767742397288,"supabase_id":"","distance":-0.07328087091445923}]}% 
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
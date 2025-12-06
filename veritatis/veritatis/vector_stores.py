from pymilvus import (
   connections,
   FieldSchema, CollectionSchema, DataType, Collection, utility
)
from dotenv import load_dotenv
import os

load_dotenv()
# Connect to Milvus
def connect_milvus():
   connections.connect(
      alias="default",
      host=os.getenv("MILVUS_HOST", "localhost"),
      port=os.getenv("MILVUS_PORT", "19530"),
   )
   print("✅ Connected to Milvus!")

def create_collection_if_not_exists(name: str, fields, description: str, index_params: dict):
   """Create Milvus collection safely (idempotent)."""
   if utility.has_collection(name):
      print(f"Collection '{name}' already exists — skipping.")
      return Collection(name)

   schema = CollectionSchema(fields=fields, description=description)
   collection = Collection(name=name, schema=schema)
   print(f"Created collection '{name}'.")

   # Create index on embedding field
   collection.create_index(field_name="embedding", index_params=index_params)
   print(f"Index created for '{name}': {index_params}")

   return collection
def init_collections():
   """Initialize all three Veritatis tiers."""
   connect_milvus()

	# Tier 1: Lacus Factorum
   tier1_fields = [
		FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
		FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
		FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
		FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=500),
		FieldSchema(name="credibility_score", dtype=DataType.FLOAT),
		FieldSchema(name="ingested_timestamp", dtype=DataType.INT64),
	]
   tier1_index = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}

   create_collection_if_not_exists(
		"veritatis_tier1_lake", tier1_fields, "Lacus Factorum — unverified facts", tier1_index
	)

	# Tier 2: Arena Veritatis
   tier2_fields = [
		FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
		FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
		FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
		FieldSchema(name="verification_confidence", dtype=DataType.FLOAT),
		FieldSchema(name="cross_source_count", dtype=DataType.INT64),
	]
   tier2_index = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 200}}

   create_collection_if_not_exists(
		"veritatis_tier2_arena", tier2_fields, "Arena Veritatis — candidate facts", tier2_index
	)

	# Tier 3: Sanctum Veritatis
   tier3_fields = [
		FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
		FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
		FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
		FieldSchema(name="verified_by", dtype=DataType.VARCHAR, max_length=200),
		FieldSchema(name="last_review_timestamp", dtype=DataType.INT64),
	]
   tier3_index = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 300}}

   create_collection_if_not_exists(
		"veritatis_tier3_sanctum", tier3_fields, "Sanctum Veritatis — verified facts", tier3_index
	)

   print("✅ All collections initialized.")
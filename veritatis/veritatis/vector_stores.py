"""Milvus vector store setup for Veritatis."""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from pymilvus import (
   Collection,
   CollectionSchema,
   DataType,
   FieldSchema,
   MilvusClient,
   connections,
   utility,
)

ENV = os.getenv("ENV", "development")
if ENV == "development":
   load_dotenv()

MILVUS_SKIP_CONNECT = os.getenv("MILVUS_SKIP_CONNECT", "false").lower() == "true"
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

def ensure_connection() -> None:
   """Establish a connection to Milvus if not already connected and not skipped.
   Avoids importing-time connection attempts which crash when Milvus isn't ready.
   """
   if MILVUS_SKIP_CONNECT:
      return
   try:
      connections.connect(
         alias="default",
         host=MILVUS_HOST,
         port=MILVUS_PORT,
         timeout=10,
      )
   except Exception:
      raise

class MilvusRecordStore:
   def __init__(self, uri=None, host=None, port=None, client=None):
      # Initialize connection lazily to avoid import-time failures
      ensure_connection()
      if uri is None:
         host = host or MILVUS_HOST
         port = port or MILVUS_PORT
         uri = f"http://{host}:{port}"

      if client is not None:
         self.client = client
      elif os.getenv("MILVUS_SKIP_CONNECT") == "1":
         class _Dummy:
            @staticmethod
            def get(collection_name, ids):
               return []
         self.client = _Dummy()
      else:
         self.client = MilvusClient(uri=uri)

   def insert_record(self, collection_name: str, record: Dict[str, Any] | list[Dict[str, Any]]):
      """Insert one or more records into the given collection.

      - Accepts a single dict or a list of dicts.
      - Validates required fields based on the collection schema.
      - Builds columnar data for Milvus (one list per field).
      Returns the primary keys generated/inserted.
      """
      collection = Collection(collection_name)
      schema_field_names = [f.name for f in collection.schema.fields]

      rows: list[Dict[str, Any]] = record if isinstance(record, list) else [record]

      for i, r in enumerate(rows):
         missing = [name for name in schema_field_names if name not in r]
         if missing:
            raise ValueError(f"Row {i} missing required fields for '{collection_name}': {missing}")

      data_columns = []
      for name in schema_field_names:
         column = [r[name] for r in rows]
         data_columns.append(column)

      res = collection.insert(data_columns)
      collection.flush()  # Ensure data is persisted and available for queries
      print(f"Inserted {len(rows)} record(s) into {collection_name}; primary_keys={getattr(res, 'primary_keys', None)}")
      return getattr(res, 'primary_keys', None)

   def record_exists(self, collection_name: str, record_id: str) -> bool:
      """Return True if a record with primary key id exists in the collection."""
      collection = Collection(collection_name)
      results = collection.query(expr=f'id in ["{record_id}"]', output_fields=['id'])
      return len(results) > 0

   def delete_record(self, collection_name:str, record_id: str):
      collection = Collection(collection_name)
      # Use 'in' expression which is more reliable for VARCHAR primary keys
      result = collection.delete(expr=f'id in ["{record_id}"]')
      collection.flush()  # Ensure deletion is persisted
      # Force compaction to immediately remove deleted records (for testing)
      import time
      time.sleep(0.1)  # Brief wait for flush to complete
      print(f"Deleted record {record_id} from {collection_name}; delete_count={getattr(result, 'delete_count', 'unknown')}")

   def get_record(self, collection_name: str, record_id: str):
      # Query with explicit field names including the vector field
      try:
         collection = Collection(collection_name)
         # Get all field names from schema
         field_names = [f.name for f in collection.schema.fields]
         results = collection.query(expr=f'id in ["{record_id}"]', output_fields=field_names)
      except Exception as e:
         print(f"Error fetching record {record_id} from {collection_name}: {e}")
         return None
      return results[0] if results else None
   
   def move_record(self, collection_from: str, collection_to: str, record_id: str):
      record = self.get_record(collection_from, record_id)
      if not record:
         print(f"No such record found in {collection_from}")
         return False
      # Ensure the target collection has all required fields; missing optional fields filled if needed.
      self.insert_record(collection_to, record)
      self.delete_record(collection_from, record_id)
      print(f"Record id={record_id} moved {collection_from} -> {collection_to}")
      return True

def create_collection_if_not_exists(
    name: str, fields, description: str, index_params: dict
):
    """Create Milvus collection safely (idempotent)."""
    if utility.has_collection(name):
        print(f"Collection '{name}' already exists — loading into memory.")
        collection = Collection(name)
        collection.load()
        print(f"Collection '{name}' loaded.")
        return collection

    schema = CollectionSchema(fields=fields, description=description)
    collection = Collection(name=name, schema=schema)
    print(f"Created collection '{name}'.")

    # Create index on embedding field
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Index created for '{name}': {index_params}")

    # Load collection into memory for querying
    collection.load()
    print(f"Collection '{name}' loaded into memory.")

    return collection


def init_collections():
    """Initialize all three Veritatis tiers."""

    # Tier 1: Lacus Factorum
    tier1_fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="credibility_score", dtype=DataType.FLOAT),
        FieldSchema(name="ingested_timestamp", dtype=DataType.INT64),
        FieldSchema(name="supabase_id", dtype=DataType.VARCHAR, max_length=100),
    ]
    tier1_index = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 32, "efConstruction": 200},
    }

    create_collection_if_not_exists(
        "veritatis_tier1_lake",
        tier1_fields,
        "Lacus Factorum — unverified facts",
        tier1_index,
    )

    # Tier 2: Arena Veritatis
    tier2_fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="verification_confidence", dtype=DataType.FLOAT),
        FieldSchema(name="cross_source_count", dtype=DataType.INT64),
        FieldSchema(name="supabase_id", dtype=DataType.VARCHAR, max_length=100),
    ]
    tier2_index = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 32, "efConstruction": 300},
    }

    create_collection_if_not_exists(
        "veritatis_tier2_arena",
        tier2_fields,
        "Arena Veritatis — candidate facts",
        tier2_index,
    )

    # Tier 3: Sanctum Veritatis
    tier3_fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="verified_by", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="last_review_timestamp", dtype=DataType.INT64),
        FieldSchema(name="supabase_id", dtype=DataType.VARCHAR, max_length=100),
    ]
    tier3_index = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 32, "efConstruction": 400},
    }

    create_collection_if_not_exists(
        "veritatis_tier3_sanctum",
        tier3_fields,
        "Sanctum Veritatis — verified facts",
        tier3_index,
    )

    print("✅ All collections initialized.")
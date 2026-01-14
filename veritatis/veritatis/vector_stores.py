"""Milvus vector store setup for Veritatis."""

import logging
import math
import os
import time
from typing import Any, Dict, Iterable, Optional, Union

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
from pymilvus.client.types import LoadState

logger = logging.getLogger(__name__)

ENV = os.getenv("ENV", "development")
if ENV == "development":
    load_dotenv()

MILVUS_RECREATE_ON_STARTUP = (
    os.getenv("MILVUS_RECREATE_ON_STARTUP", "false").lower() == "true"
)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))


def _field_signature(field: FieldSchema) -> Dict[str, Any]:
    """Return a stable, comparable signature for a Milvus field."""
    sig: Dict[str, Any] = {
        "name": field.name,
        "dtype": int(field.dtype),
        "is_primary": bool(getattr(field, "is_primary", False)),
    }
    # Optional attributes depend on dtype
    if hasattr(field, "dim") and getattr(field, "dim", None) is not None:
        sig["dim"] = int(field.dim)
    if hasattr(field, "max_length") and getattr(field, "max_length", None) is not None:
        sig["max_length"] = int(field.max_length)
    return sig


def _schema_signature(fields: Iterable[FieldSchema]) -> list[Dict[str, Any]]:
    """Build a stable schema signature for comparison/debugging."""
    return [_field_signature(f) for f in fields]


def _ensure_collection_schema_matches(
    name: str, expected_fields: Iterable[FieldSchema]
) -> None:
    """Fail fast if an existing collection schema doesn't match what code expects.

    This catches the common case where old on-disk Milvus state was created with a
    different schema (e.g. different dim/max_length/field set), which can lead to
    confusing runtime failures and sometimes even server-side instability.
    """
    existing = Collection(name)
    existing_sig = _schema_signature(existing.schema.fields)
    expected_sig = _schema_signature(list(expected_fields))
    if existing_sig != expected_sig:
        raise ValueError(
            "Existing collection schema does not match expected schema. "
            f"collection={name}. "
            "Volumes may contain old state created by a different schema. "
            "Fix by wiping volumes or setting MILVUS_RECREATE_ON_STARTUP=true. "
            f"existing={existing_sig} expected={expected_sig}"
        )


def _validate_scalar_value(
    name: str, dtype: DataType, value: Any, *, max_length: Optional[int] = None
) -> None:
    """Validate a scalar field value matches the expected Milvus dtype."""
    if dtype == DataType.VARCHAR:
        if not isinstance(value, str):
            raise TypeError(
                f"Field '{name}' must be str (VARCHAR), got {type(value).__name__}"
            )
        if max_length is not None and len(value) > max_length:
            raise ValueError(
                f"Field '{name}' exceeds max_length={max_length} (got {len(value)})"
            )
        return

    if dtype == DataType.INT64:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"Field '{name}' must be int (INT64), got {type(value).__name__}"
            )
        return

    if dtype == DataType.FLOAT:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Field '{name}' must be float (FLOAT), got {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"Field '{name}' must be finite (no NaN/Inf)")
        return


def _validate_vector_value(name: str, value: Any, *, dim: int) -> None:
    """Validate a vector field is finite numeric and has the expected dimension."""
    # Accept list/tuple and numpy-like arrays.
    try:
        length = len(value)
    except Exception:
        raise TypeError(f"Field '{name}' must be a vector (sequence) of length {dim}")

    if length != dim:
        raise ValueError(
            f"Field '{name}' vector dim mismatch: expected {dim}, got {length}"
        )

    # Ensure all values are finite numbers (NaNs/Infs can cause downstream issues)
    for i, x in enumerate(value):
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise TypeError(
                f"Field '{name}' vector element {i} must be float, "
                f"got {type(x).__name__}"
            )
        if not math.isfinite(float(x)):
            raise ValueError(
                f"Field '{name}' vector element {i} is not finite (NaN/Inf)"
            )


def validate_record_against_collection_schema(
    collection_name: str, row: Dict[str, Any]
) -> None:
    """Validate one record dict against the live Milvus schema for the collection."""
    collection = Collection(collection_name)
    for field in collection.schema.fields:
        # Milvus requires all fields we insert; we already enforce presence elsewhere.
        value = row[field.name]
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = int(field.params.get("dim") or getattr(field, "dim", 0))
            _validate_vector_value(
                field.name,
                value,
                dim=dim,
            )
        else:
            _validate_scalar_value(
                field.name,
                field.dtype,
                value,
                max_length=getattr(field, "max_length", None),
            )


def ensure_connection() -> None:
    """Establish a connection to Milvus if not already connected.

    Avoids importing-time connection attempts which crash when Milvus isn't ready.
    """
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            timeout=10,
        )
    except Exception:
        raise

    timeout = 60
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # This will throw if Milvus isn't ready
            utility.list_collections()
            logger.info("✅ Milvus is ready")
            break
        except Exception:
            logger.debug("Milvus not ready yet, waiting...")
            time.sleep(0.5)
    else:
        raise TimeoutError("Milvus did not become ready within 60 seconds")


def wait_for_collection_ready(name, timeout=60):
    """Wait until Milvus reports the collection as either Loaded or NotLoad."""
    start = time.time()
    last_state = None
    while time.time() - start < timeout:
        try:
            state = utility.load_state(name)
            if state in (LoadState.Loaded, LoadState.NotLoad):
                return state
            last_state = state
            # state is Loading → sleep briefly and check again
            time.sleep(0.1)
        except Exception:
            # Sometimes Milvus isn’t fully initialized yet
            time.sleep(0.1)
    raise TimeoutError(f"Collection {name} never became ready (state={last_state})")


def ensure_collection_loaded(collection_name: str) -> None:
    """Ensure a collection is loaded into memory before insert/search operations."""
    try:
        load_state = utility.load_state(collection_name)
        if load_state == LoadState.Loaded:
            return  # Already loaded

        logger.info(f"📥 Auto-loading collection '{collection_name}' for first use...")

        # Prefer Collection API for consistency.
        collection = Collection(collection_name)
        # pymilvus supports timeout kwarg; if server gets stuck, we want to surface it.
        collection.load(timeout=30)

        # Verify loaded state
        if utility.load_state(collection_name) != LoadState.Loaded:
            raise TimeoutError(
                f"Collection '{collection_name}' did not reach Loaded state"
            )
        logger.info(f"✅ Collection '{collection_name}' loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load collection '{collection_name}': {e}")
        raise


class MilvusRecordStore:
    """Manage records in Milvus collections."""

    def __init__(self, uri=None, host=None, port=None, client=None):
        """Create a record store wrapper and connect to Milvus lazily."""
        # Initialize connection lazily to avoid import-time failures
        ensure_connection()
        if uri is None:
            host = host or MILVUS_HOST
            port = port or MILVUS_PORT
            uri = f"http://{host}:{port}"

        if client is not None:
            self.client = client
        else:
            self.client = MilvusClient(uri=uri)

    def insert_record(
        self,
        collection_name: str,
        record: Union[Dict[str, Any], list[Dict[str, Any]]],
    ):
        """Insert one or more records into the given collection.

        - Accepts a single dict or a list of dicts.
        - Validates required fields based on the collection schema.
        - Builds columnar data for Milvus (one list per field).
        Returns the primary keys generated/inserted.
        """
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        schema_field_names = [f.name for f in collection.schema.fields]

        rows: list[Dict[str, Any]] = record if isinstance(record, list) else [record]

        for i, r in enumerate(rows):
            missing = [name for name in schema_field_names if name not in r]
            if missing:
                raise ValueError(
                    f"Row {i} missing required fields for '{collection_name}': "
                    f"{missing}"
                )
            # Catch type/dimension problems early (common source of "bad" entities)
            validate_record_against_collection_schema(collection_name, r)

        data_columns = []
        for name in schema_field_names:
            column = [r[name] for r in rows]
            data_columns.append(column)

        res = collection.insert(data_columns)
        collection.flush()  # Ensure data is persisted and available for queries
        print(
            f"Inserted {len(rows)} record(s) into {collection_name}; "
            f"primary_keys={getattr(res, 'primary_keys', None)}"
        )
        return getattr(res, "primary_keys", None)

    def record_exists(self, collection_name: str, record_id: str) -> bool:
        """Return True if a record with primary key id exists in the collection."""
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        results = collection.query(
            expr=f'id in ["{record_id}"]',
            output_fields=["id"],
        )
        return len(results) > 0

    def delete_record(self, collection_name: str, record_id: str):
        """Delete a record from a collection."""
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        # Use 'in' expression which is more reliable for VARCHAR primary keys
        result = collection.delete(expr=f'id in ["{record_id}"]')
        collection.flush()  # Ensure deletion is persisted
        # Force compaction to immediately remove deleted records (for testing)
        import time

        time.sleep(0.1)  # Brief wait for flush to complete
        print(
            f"Deleted record {record_id} from {collection_name}; "
            f"delete_count={getattr(result, 'delete_count', 'unknown')}"
        )

    def get_record(self, collection_name: str, record_id: str):
        """Retrieve a record from a collection."""
        # Query with explicit field names including the vector field
        try:
            ensure_collection_loaded(collection_name)
            collection = Collection(collection_name)
            # Get all field names from schema
            field_names = [f.name for f in collection.schema.fields]
            results = collection.query(
                expr=f'id in ["{record_id}"]', output_fields=field_names
            )
        except Exception as e:
            print(f"Error fetching record {record_id} from {collection_name}: {e}")
            return None
        return results[0] if results else None

    def move_record(self, collection_from: str, collection_to: str, record_id: str):
        """Move a record between collections."""
        record = self.get_record(collection_from, record_id)
        if not record:
            print(f"No such record found in {collection_from}")
            return False
        # Ensure the target collection has all required fields.
        self.insert_record(collection_to, record)
        self.delete_record(collection_from, record_id)
        print(f"Record id={record_id} moved {collection_from} -> {collection_to}")
        return True


def create_collection_if_not_exists(
    name: str, fields, description: str, index_params: dict
):
    """Create Milvus collections."""
    if utility.has_collection(name):
        if MILVUS_RECREATE_ON_STARTUP:
            logger.warning(
                f"🧨 Dropping existing collection '{name}' due to "
                "MILVUS_RECREATE_ON_STARTUP=true"
            )
            utility.drop_collection(name)
        else:
            # Validate schema matches what this code expects.
            _ensure_collection_schema_matches(name, fields)
            logger.info(
                f"⚡ Collection '{name}' already exists (schema OK; "
                "will auto-load on first use)"
            )
            return Collection(name)

    # Create new collection
    logger.info(f"🆕 Creating new collection '{name}'...")
    start_total = time.time()

    schema = CollectionSchema(fields=fields, description=description)
    collection = Collection(name=name, schema=schema)
    logger.info(f"✅ Collection '{name}' created")

    # Create index
    logger.info(f"🔧 Creating index for '{name}'...")
    start_index = time.time()
    collection.create_index(field_name="embedding", index_params=index_params)
    elapsed_index = time.time() - start_index
    logger.info(f"✅ Index created in {elapsed_index:.2f}s")

    # Load using client API (more reliable than collection.load())
    logger.info(f"📥 Loading '{name}' into memory...")
    start_load = time.time()
    # Keep Collection API for consistency.
    collection.load(timeout=30)

    # Wait for load to complete
    timeout_end = time.time() + 30
    while time.time() < timeout_end:
        if utility.load_state(name) == LoadState.Loaded:
            break
        time.sleep(0.5)

    elapsed_load = time.time() - start_load
    logger.info(f"✅ Collection loaded in {elapsed_load:.2f}s")

    total_elapsed = time.time() - start_total
    logger.info(f"🎯 Collection '{name}' ready in {total_elapsed:.2f}s")

    return collection


def init_collections():
    """Initialize Veritatis collections (Tier 1 and Tier 2)."""
    logger.info("Initializing all Milvus collections...")
    start_all = time.time()

    # Tier 1: Lacus Factorum
    logger.info("Tier 1: Lacus Factorum (veritatis_tier1_lake)")
    tier1_fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        # Store JSON-serialized metadata as VARCHAR for compatibility.
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
    ]
    tier1_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 200},
    }

    create_collection_if_not_exists(
        "veritatis_tier1_lake",
        tier1_fields,
        "Lacus Factorum — unverified facts",
        tier1_index,
    )

    # Tier 2: Arena Veritatis
    logger.info("Tier 2: Arena Veritatis (veritatis_tier2_arena)")
    tier2_fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="verification_confidence", dtype=DataType.FLOAT),
        FieldSchema(name="cross_source_count", dtype=DataType.INT64),
    ]
    tier2_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 300},
    }

    create_collection_if_not_exists(
        "veritatis_tier2_arena",
        tier2_fields,
        "Arena Veritatis — candidate facts",
        tier2_index,
    )
    total_elapsed = time.time() - start_all
    logger.info(f"✅ Collections initialized in {total_elapsed:.2f}s")

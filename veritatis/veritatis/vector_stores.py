"""Milvus vector store setup for Veritatis."""

import logging
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Union

from dotenv import load_dotenv
from pymilvus import (  # noqa: E402
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from pymilvus.client.types import LoadState  # noqa: E402

logger = logging.getLogger(__name__)

ENV = os.getenv("ENV", "development")
if ENV == "development":
    load_dotenv()


MILVUS_SKIP_CONNECT = os.getenv("MILVUS_SKIP_CONNECT", "false").lower() == "true"
MILVUS_RECREATE_ON_STARTUP = (
    os.getenv("MILVUS_RECREATE_ON_STARTUP", "true").lower() == "true"
)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# One collection per credibility tier.
TIER1_COLLECTION = "veritatis_tier1"  # raw / unverified
TIER2_COLLECTION = "veritatis_tier2"  # credible (score > 0.7)
TIER3_COLLECTION = "veritatis_tier3"  # verified

ALL_TIER_COLLECTIONS = [TIER1_COLLECTION, TIER2_COLLECTION, TIER3_COLLECTION]

# Legacy alias kept so existing imports don't break immediately.
COLLECTION_NAME = TIER1_COLLECTION


def collection_for_tier(tier: int) -> str:
    """Return the collection name for a given tier (1, 2, or 3)."""
    if tier < 1 or tier > 3:
        raise ValueError(f"Invalid tier {tier}; must be 1, 2, or 3")
    return ALL_TIER_COLLECTIONS[tier - 1]


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


def _schema_signature(fields: Iterable[FieldSchema]) -> List[Dict[str, Any]]:
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
            "Existing Milvus collection schema does not match expected "
            f"schema. collection={name}. This usually means your Milvus "
            "volumes contain old state created by a different schema. "
            "Fix by wiping volumes or setting "
            f"MILVUS_RECREATE_ON_STARTUP=true. existing={existing_sig} "
            f"expected={expected_sig}"
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
            _validate_vector_value(
                field.name,
                value,
                dim=int(field.params.get("dim") or getattr(field, "dim", 0)),
            )
        else:
            _validate_scalar_value(
                field.name,
                field.dtype,
                value,
                max_length=getattr(field, "max_length", None),
            )


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
            # Sometimes Milvus isn't fully initialized yet
            time.sleep(0.1)
    raise TimeoutError(f"Collection {name} never became ready (state={last_state})")


def ensure_collection_loaded(collection_name: str) -> None:
    """
    Ensure a collection is loaded into memory before operations.

    Loads collection if not already loaded for insert/search operations.
    """
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
                f"Collection '{collection_name}' did not reach Loaded " "state"
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
        elif os.getenv("MILVUS_SKIP_CONNECT") == "1":

            class _Dummy:
                @staticmethod
                def get(collection_name, ids):
                    return []

            self.client = _Dummy()
        else:
            self.client = MilvusClient(uri=uri)

    def insert_record(
        self,
        collection_name: str,
        record: Union[Dict[str, Any], List[Dict[str, Any]]],
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

        rows: List[Dict[str, Any]] = record if isinstance(record, list) else [record]

        for i, r in enumerate(rows):
            missing = [name for name in schema_field_names if name not in r]
            if missing:
                raise ValueError(
                    f"Row {i} missing required fields for "
                    f"'{collection_name}': {missing}"
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

    def record_exists(self, collection_name: str, record_iid: str) -> bool:
        """Return True if a record with primary key iid exists in the collection."""
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        results = collection.query(
            expr=f'iid in ["{record_iid}"]', output_fields=["iid"]
        )
        return len(results) > 0

    def delete_record(self, collection_name: str, record_iid: str):
        """Delete a record from a collection."""
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        result = collection.delete(expr=f'iid in ["{record_iid}"]')
        collection.flush()
        time.sleep(0.1)
        print(
            f"Deleted record {record_iid} from {collection_name}; "
            f"delete_count={getattr(result, 'delete_count', 'unknown')}"
        )

    def get_record(self, collection_name: str, record_iid: str):
        """Retrieve a record from a collection by iid."""
        try:
            ensure_collection_loaded(collection_name)
            collection = Collection(collection_name)
            field_names = [f.name for f in collection.schema.fields]
            results = collection.query(
                expr=f'iid in ["{record_iid}"]', output_fields=field_names
            )
        except Exception as e:
            print(f"Error fetching record {record_iid} from {collection_name}: {e}")
            return None
        return results[0] if results else None

    def move_records(
        self, source_collection: str, target_collection: str, iids: List[str]
    ) -> int:
        """Move records from one tier collection to another.

        Fetches full records from *source_collection*, deletes them there,
        and re-inserts into *target_collection*.
        Returns the number of records moved.
        """
        ensure_collection_loaded(source_collection)
        ensure_collection_loaded(target_collection)

        src = Collection(source_collection)
        iid_list = ", ".join(f'"{iid}"' for iid in iids)
        field_names = [f.name for f in src.schema.fields]
        records = src.query(expr=f"iid in [{iid_list}]", output_fields=field_names)
        if not records:
            return 0

        # Delete from source
        src.delete(expr=f"iid in [{iid_list}]")
        src.flush()
        time.sleep(0.1)

        # Insert into target
        tgt = Collection(target_collection)
        tgt_field_names = [f.name for f in tgt.schema.fields]
        data_columns = [[r[name] for r in records] for name in tgt_field_names]
        tgt.insert(data_columns)
        tgt.flush()
        return len(records)

    def update_credibility_scores(
        self, collection_name: str, updates: List[Dict[str, Any]]
    ) -> int:
        """Update credibility_score for records.

        updates: list of {"iid": str, "credibility_score": float}
        Returns the number of records updated.
        """
        if not updates:
            return 0
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        iids = [u["iid"] for u in updates]
        score_map = {u["iid"]: u["credibility_score"] for u in updates}
        iid_list = ", ".join(f'"{iid}"' for iid in iids)
        field_names = [f.name for f in collection.schema.fields]
        records = collection.query(
            expr=f"iid in [{iid_list}]", output_fields=field_names
        )
        if not records:
            return 0

        for record in records:
            record["credibility_score"] = score_map.get(
                record["iid"], record["credibility_score"]
            )

        collection.delete(expr=f"iid in [{iid_list}]")
        collection.flush()
        time.sleep(0.1)

        schema_field_names = [f.name for f in collection.schema.fields]
        data_columns = [[r[name] for r in records] for name in schema_field_names]
        collection.insert(data_columns)
        collection.flush()
        return len(records)

    def query_by_tier(
        self,
        tier: int,
        *,
        min_credibility: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query records from the collection corresponding to *tier*.

        Optionally filters by minimum credibility_score.
        """
        collection_name = collection_for_tier(tier)
        ensure_collection_loaded(collection_name)
        collection = Collection(collection_name)
        expr = ""
        if min_credibility is not None:
            expr = f"credibility_score >= {min_credibility}"
        field_names = [f.name for f in collection.schema.fields]
        results: list[dict[str, Any]] = collection.query(
            expr=expr or None, output_fields=field_names, limit=limit
        )
        return results


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
                f"⚡ Collection '{name}' already exists (schema OK; will "
                "auto-load on first use)"
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
    """Initialize the three Veritatis tier collections."""
    logger.info("Initializing Veritatis Milvus collections...")
    start_all = time.time()

    fields = [
        FieldSchema(
            name="iid",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
            description="UUID from parsed_content.id",
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="credibility_score", dtype=DataType.FLOAT),
        FieldSchema(
            name="date",
            dtype=DataType.INT64,
            description="Earthquake event date as epoch milliseconds",
        ),
        FieldSchema(
            name="domain",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="Source domain from page_schemas",
        ),
    ]

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 200},
    }

    descriptions = {
        TIER1_COLLECTION: "Veritatis tier 1 — raw / unverified earthquake embeddings",
        TIER2_COLLECTION: "Veritatis tier 2 — credible earthquake embeddings",
        TIER3_COLLECTION: "Veritatis tier 3 — verified earthquake embeddings",
    }

    for col_name in ALL_TIER_COLLECTIONS:
        create_collection_if_not_exists(
            col_name,
            fields,
            descriptions[col_name],
            index_params,
        )

    total_elapsed = time.time() - start_all
    logger.info(f"✅ Veritatis collections initialized in {total_elapsed:.2f}s")

"""Pytest configuration and shared test setup."""

# Ensure project root (containing the 'veritatis' package) is on sys.path for tests.
import logging
import os
import sys
from pathlib import Path

import pytest  # noqa: E402
from pymilvus import DataType, FieldSchema, connections, utility  # noqa: E402

from veritatis.vector_stores import (  # noqa: E402
    MilvusRecordStore,
    create_collection_if_not_exists,
    ensure_connection,
)

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable to skip connection for unit tests that don't need Milvus
os.environ.setdefault("MILVUS_SKIP_CONNECT", "false")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring Milvus"
    )


@pytest.fixture(scope="session")
def milvus_connection():
    """Establish Milvus connection for integration tests."""
    try:
        ensure_connection()
        logger.info("✅ Milvus connection established for tests")
        yield
        # Cleanup after all tests
        try:
            connections.disconnect("default")
            logger.info("🔌 Milvus connection closed")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {e}")
        pytest.skip(f"Milvus not available: {e}")


@pytest.fixture(scope="function")
def test_collection(milvus_connection):
    """
    Create a temporary test collection for integration tests.

    Uses the same schema as the production ``veritatis`` collection.
    Automatically cleans up after each test.
    """
    collection_name = "test_veritatis_collection"

    # Clean up any existing test collection
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info(f"🧹 Dropped existing test collection '{collection_name}'")

    # Schema mirrors the single veritatis collection
    fields = [
        FieldSchema(
            name="iid",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="tier", dtype=DataType.INT64),
        FieldSchema(name="credibility_score", dtype=DataType.FLOAT),
        FieldSchema(name="date", dtype=DataType.INT64),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=500),
    ]

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 100,
        },  # Smaller params for faster test setup
    }

    # Create test collection
    logger.info(f"🆕 Creating test collection '{collection_name}'")
    create_collection_if_not_exists(
        collection_name,
        fields,
        "Test collection for veritatis integration tests",
        index_params,
    )

    logger.info(f"✅ Test collection '{collection_name}' ready")

    yield collection_name

    # Cleanup after test
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"🧹 Cleaned up test collection '{collection_name}'")
    except Exception as e:
        logger.warning(f"Error cleaning up test collection: {e}")


@pytest.fixture
def record_store(milvus_connection):
    """Provide a MilvusRecordStore instance for tests."""
    return MilvusRecordStore()

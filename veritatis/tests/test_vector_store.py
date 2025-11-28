import os
import pytest

# Skip this integration test module by default; Milvus unstable on local env.
os.environ["MILVUS_SKIP_CONNECT"] = "1"
pytest.skip("Integration test requires running Milvus; skipped by default.", allow_module_level=True)

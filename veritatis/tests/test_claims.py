"""Parametrized pytest tests for the fact-checker against claims_dataset.json.

These are integration tests: they call the live LLM server and Milvus.
Run with::

    pytest tests/test_claims.py -m integration -v

They are skipped automatically when the LLM server or Milvus is unavailable.
"""

import json
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).resolve().parent / "claims_dataset.json"


def _load_dataset():
    if not DATASET_PATH.exists():
        return []
    with DATASET_PATH.open() as fh:
        return json.load(fh)


def _case_id(case):
    label = case.get("label", "?")
    claim = case.get("claim", "")[:60]
    return f"[{label}] {claim}"


ALL_CASES = _load_dataset()


@pytest.fixture(scope="module", autouse=True)
def require_services(milvus_connection):
    """Skip the entire module if Milvus is unavailable."""
    try:
        from veritatis.agent import _get_llm

        _get_llm()
    except Exception as exc:
        pytest.skip(f"LLM server not available: {exc}")


@pytest.mark.integration
@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_claim_score_in_expected_range(case):
    """Verify that fact_check() returns a score within [expected_score_min, expected_score_max]."""
    from veritatis.agent import fact_check

    claim = case["claim"]
    label = case["label"]
    score_min = float(case["expected_score_min"])
    score_max = float(case["expected_score_max"])

    result = fact_check(claim)

    assert 0.0 <= result.score <= 1.0, (
        f"Score {result.score} out of [0, 1] for claim: {claim!r}"
    )
    assert score_min <= result.score <= score_max, (
        f"[{label}] score={result.score:.3f} not in [{score_min}, {score_max}]\n"
        f"claim:     {claim!r}\n"
        f"reasoning: {result.reasoning}"
    )

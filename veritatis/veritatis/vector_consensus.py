"""Vector consensus analysis for Veritatis.

Milvus collections in Veritatis store only:
- ``iid`` (primary key; maps to ``parsed_content.id`` in Supabase)
- ``embedding``
- ``credibility_score``
- ``date``
- ``domain``

This module finds the most relevant/detailed vector from a set of vectors
by comparing them against each other (without a query). It combines:
- Centrality score: how close a vector is to all others (consensus)
- Detail score: combined length + lexical diversity metric
- Credibility score: source credibility (optional, default weight 0.0)

For the detail score, this module optionally fetches ``parsed_content.main_text``
from Supabase using ``iid`` as the row id.
"""

import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymilvus import Collection

from veritatis.vector_stores import ensure_collection_loaded

logger = logging.getLogger(__name__)


@dataclass
class VectorAnalysis:
    """Analysis result for a single vector."""

    iid: str
    main_text: str
    credibility_score: float
    date: int
    domain: str
    embedding: List[float]

    # Computed scores
    centrality_score: float  # Average similarity to all other vectors
    detail_score: float  # Normalized text length score
    combined_score: float  # Weighted combination

    # Metadata
    avg_similarity_to_others: float
    main_text_length: int


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
_SUPABASE_CLIENT: Any = None


def _get_supabase_client():
    """Return a cached Supabase client.

    Prefers reusing the shared connector from ``earthquakes_parser`` when importable.
    Falls back to ``supabase.create_client`` when available.
    """
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Missing SUPABASE_URL and Supabase key "
            "(SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY)"
        )

    try:
        from earthquakes_parser.storage.supabase import SupabaseDB

        _SUPABASE_CLIENT = SupabaseDB(url=SUPABASE_URL, key=SUPABASE_KEY).client
        return _SUPABASE_CLIENT
    except Exception:
        supabase_mod = importlib.import_module("supabase")
        _SUPABASE_CLIENT = supabase_mod.create_client(SUPABASE_URL, SUPABASE_KEY)
        return _SUPABASE_CLIENT


def _normalize_main_text(value: Any) -> str:
    """Convert Supabase main_text into a plain string suitable for scoring."""
    if value is None:
        return ""

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                decoded = json.loads(s)
                return _normalize_main_text(decoded)
            except Exception:
                return s
        return s

    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            item_s = _normalize_main_text(item)
            if item_s:
                parts.append(item_s)
        return "\n".join(parts)

    if isinstance(value, dict):
        for key in ("main_text", "text", "content"):
            if key in value:
                return _normalize_main_text(value.get(key))
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    return str(value)


def _fetch_supabase_main_text_by_iids(iids: List[str]) -> Dict[str, str]:
    """Fetch ``parsed_content.main_text`` for a list of Supabase UUIDs."""
    if not iids:
        return {}

    # If Supabase isn't configured, just return empty content.
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning(
            "SUPABASE_URL/SUPABASE_KEY not set; detail score will use empty text"
        )
        return {}

    client = _get_supabase_client()
    out: Dict[str, str] = {}

    # PostgREST has practical limits for `in_` lists; chunk defensively.
    chunk_size = 200
    for i in range(0, len(iids), chunk_size):
        chunk = iids[i : i + chunk_size]
        try:
            resp = (
                client.table("parsed_content")
                .select("id, main_text")
                .in_("id", chunk)
                .execute()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch parsed_content.main_text: {e}") from e

        rows = getattr(resp, "data", None) or []
        for row in rows:
            try:
                rid = str(row.get("id", ""))
            except (AttributeError, TypeError):
                # Skip rows where .get() fails (not a dict-like object)
                continue
            if not rid:
                continue
            out[rid] = _normalize_main_text(row.get("main_text"))
    return out


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector (normalized)
        vec2: Second vector (normalized)

    Returns:
        Similarity score in range [0, 1] (higher = more similar)
    """
    # Since vectors are L2-normalized, dot product = cosine similarity
    similarity = float(np.dot(vec1, vec2))
    # Clamp to [0, 1] range (normalized vectors should already be in this range)
    return max(0.0, min(1.0, similarity))


def compute_pairwise_similarities(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarities between all embeddings.

    Vectors are assumed to be L2-normalised, so dot product == cosine similarity.

    Args:
        embeddings: List of L2-normalised embedding vectors

    Returns:
        NxN similarity matrix where element [i,j] is cosine similarity between i and j
    """
    matrix = np.array(embeddings)  # (N, D)
    similarity_matrix = np.dot(matrix, matrix.T)  # (N, N)
    np.fill_diagonal(similarity_matrix, 1.0)  # self-similarity is always 1.0
    return np.clip(similarity_matrix, 0.0, 1.0)


def calculate_centrality_scores(similarity_matrix: np.ndarray) -> List[float]:
    """
    Calculate centrality score for each vector.

    Centrality = average similarity to all OTHER vectors (diagonal excluded).
    Higher score = more central/representative.

    Args:
        similarity_matrix: NxN pairwise similarity matrix

    Returns:
        List of centrality scores (one per vector)
    """
    n = similarity_matrix.shape[0]
    if n == 1:
        return [1.0]

    row_sums = similarity_matrix.sum(axis=1) - 1.0  # exclude self-similarity diagonal
    centrality = row_sums / (n - 1)
    return [float(x) for x in centrality]


def calculate_detail_scores(contents: List[str]) -> List[float]:
    """
    Calculate detail scores using a combined length + lexical diversity metric.

    Raw score = 0.7 * normalized_length + 0.3 * lexical_diversity
    where lexical_diversity = unique_words / total_words (penalises repetitive text).
    The raw scores are then re-normalised to [0, 1].

    Args:
        contents: List of content strings

    Returns:
        List of detail scores in [0, 1]
    """
    if not contents:
        return []

    lengths = [len(content) for content in contents]
    min_length = min(lengths)
    max_length = max(lengths)

    if max_length == min_length:
        normalized_lengths = [1.0] * len(contents)
    else:
        normalized_lengths = [
            (length - min_length) / (max_length - min_length) for length in lengths
        ]

    def _lexical_diversity(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    diversities = [_lexical_diversity(c) for c in contents]

    raw_scores = [
        0.7 * nl + 0.3 * ld for nl, ld in zip(normalized_lengths, diversities)
    ]

    min_raw = min(raw_scores)
    max_raw = max(raw_scores)
    if max_raw == min_raw:
        return [1.0] * len(contents)

    return [(s - min_raw) / (max_raw - min_raw) for s in raw_scores]


def fetch_all_vectors(
    collection_name: str, limit: Optional[int] = None, offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Fetch all vectors from a Milvus collection.

    Attempts to retrieve the embedding field directly from Milvus via
    ``output_fields``.

    Args:
        collection_name: Name of the collection
        limit: Maximum number of vectors to fetch (None = all)
        offset: Number of vectors to skip

    Returns:
        List of vector records with all fields including embeddings
    """
    logger.info(
        f"Fetching vectors from '{collection_name}' (limit={limit}, offset={offset})"
    )

    ensure_collection_loaded(collection_name)
    collection = Collection(collection_name)

    num_entities = collection.num_entities
    logger.info(f"Collection has {num_entities} entities")

    if limit is None:
        limit = num_entities

    expr = "iid != ''"  # Match all records

    results = collection.query(
        expr=expr,
        output_fields=[
            "iid",
            "embedding",
            "credibility_score",
            "date",
            "domain",
        ],
        limit=limit,
        offset=offset,
    )

    # Check whether Milvus actually returned the embedding field
    has_embeddings = results and results[0].get("embedding") is not None

    if has_embeddings:
        logger.info(f"Fetched {len(results)} records with embeddings from Milvus")
        return [
            {
                "iid": str(record.get("iid") or ""),
                "embedding": record["embedding"],
                "credibility_score": float(record.get("credibility_score", 0.0)),
                "date": int(record.get("date", 0)),
                "domain": str(record.get("domain") or ""),
            }
            for record in results
            if record.get("iid")
        ]

    logger.error(
        "Milvus query did not return embeddings for collection '%s'. ",
        collection_name,
    )
    raise RuntimeError(
        f"Milvus did not return embeddings for collection '{collection_name}'."
    )


def find_most_relevant_vector(
    collection_name: str,
    centrality_weight: float = 0.6,
    detail_weight: float = 0.4,
    credibility_weight: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Tuple[VectorAnalysis, List[VectorAnalysis]]:
    """
    Find the most relevant vector from a collection by comparing all vectors.

    Combines up to three factors:
    1. Centrality: How similar a vector is to all others (consensus)
    2. Detail: Combined length + lexical diversity score
    3. Credibility: Source credibility score (optional, default weight 0.0)

    All three weights must sum to 1.0. Default 0.6/0.4/0.0 is fully backward
    compatible with previous two-factor behaviour.

    This is useful for filtering vectors in tier1 before promoting to tier2.
    The "best" vector represents the most central (consensus) and detailed entry.

    Args:
        collection_name: Name of the Milvus collection (e.g., "veritatis_tier1_lake")
        centrality_weight: Weight for centrality score (default: 0.6)
        detail_weight: Weight for detail score (default: 0.4)
        credibility_weight: Weight for credibility score (default: 0.0)
        limit: Maximum number of vectors to analyze (None = all)
        offset: Number of vectors to skip from start

    Returns:
        Tuple of (best_vector, all_vectors_ranked)

    Example:
        >>> best, all_ranked = find_most_relevant_vector(
        ...     "veritatis_tier1_lake",
        ...     centrality_weight=0.6,
        ...     detail_weight=0.4
        ... )
        >>> print(f"Best: {best.id}, score: {best.combined_score:.3f}")
    """
    logger.info(f"Analyzing vectors in '{collection_name}'")
    logger.info(
        f"Weights - centrality: {centrality_weight}, detail: {detail_weight}, "
        f"credibility: {credibility_weight}"
    )

    # Validate weights
    assert (
        abs(centrality_weight + detail_weight + credibility_weight - 1.0) < 1e-6
    ), "Weights must sum to 1.0"

    # Fetch all vectors with embeddings from Milvus, then fetch main_text from Supabase.
    records = fetch_all_vectors(collection_name, limit=limit, offset=offset)

    if not records:
        raise ValueError(f"No vectors found in collection '{collection_name}'")

    iids = [str(r.get("iid", "")) for r in records if r.get("iid")]
    main_text_map = _fetch_supabase_main_text_by_iids(iids)
    for r in records:
        rid = str(r.get("iid", ""))
        r["main_text"] = main_text_map.get(rid, "")

    # Use the helper function for the analysis
    return find_best_vector_with_embeddings(
        records,
        centrality_weight=centrality_weight,
        detail_weight=detail_weight,
        credibility_weight=credibility_weight,
    )


def find_best_vector_with_embeddings(
    vectors_with_embeddings: List[Dict[str, Any]],
    centrality_weight: float = 0.6,
    detail_weight: float = 0.4,
    credibility_weight: float = 0.0,
) -> Tuple[VectorAnalysis, List[VectorAnalysis]]:
    """
    Find the most relevant vector from a pre-fetched list of vectors.

    This is a convenience function for when embeddings are already fetched.

    Args:
        vectors_with_embeddings: List of dicts with fields:
                        - iid, main_text, credibility_score, date, domain, embedding
        centrality_weight: Weight for centrality score (default: 0.6)
        detail_weight: Weight for detail score (default: 0.4)
        credibility_weight: Weight for credibility score (default: 0.0)

    Returns:
        Tuple of (best_vector, all_vectors_ranked)
    """
    if not vectors_with_embeddings:
        raise ValueError("Empty vector list provided")

    # Validate weights
    assert (
        abs(centrality_weight + detail_weight + credibility_weight - 1.0) < 1e-6
    ), "Weights must sum to 1.0"

    if len(vectors_with_embeddings) == 1:
        record = vectors_with_embeddings[0]
        text = str(record.get("main_text") or "")
        analysis = VectorAnalysis(
            iid=str(record.get("iid") or ""),
            main_text=text,
            credibility_score=float(record.get("credibility_score", 0.0)),
            date=int(record.get("date", 0)),
            domain=str(record.get("domain") or ""),
            embedding=record["embedding"],
            centrality_score=1.0,
            detail_score=1.0,
            combined_score=1.0,
            avg_similarity_to_others=1.0,
            main_text_length=len(text),
        )
        return analysis, [analysis]

    # Extract data
    embeddings = [np.array(r["embedding"]) for r in vectors_with_embeddings]
    contents = [str(r.get("main_text") or "") for r in vectors_with_embeddings]
    credibility_scores = [
        float(r.get("credibility_score", 0.0)) for r in vectors_with_embeddings
    ]

    # Compute similarities
    similarity_matrix = compute_pairwise_similarities(embeddings)

    # Calculate scores
    centrality_scores = calculate_centrality_scores(similarity_matrix)
    detail_scores = calculate_detail_scores(contents)

    # Combine scores
    # credibility_score is already normalised [0, 1] (FLOAT in Milvus)
    combined_scores = [
        centrality_weight * cent + detail_weight * det + credibility_weight * cred
        for cent, det, cred in zip(centrality_scores, detail_scores, credibility_scores)
    ]

    # Create analysis objects
    analyses = []
    for i, record in enumerate(vectors_with_embeddings):
        iid = str(record.get("iid") or "")
        text = str(record.get("main_text") or "")
        analysis = VectorAnalysis(
            iid=iid,
            main_text=text,
            credibility_score=credibility_scores[i],
            date=int(record.get("date", 0)),
            domain=str(record.get("domain") or ""),
            embedding=record["embedding"],
            centrality_score=centrality_scores[i],
            detail_score=detail_scores[i],
            combined_score=combined_scores[i],
            avg_similarity_to_others=centrality_scores[i],
            main_text_length=len(text),
        )
        analyses.append(analysis)

    # Sort by combined score
    analyses.sort(key=lambda x: x.combined_score, reverse=True)

    return analyses[0], analyses

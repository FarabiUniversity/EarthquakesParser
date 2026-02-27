"""
Enhanced search logic with relevance filtering for veritatis.

This module adds relevance threshold filtering to vector search results,
keeping only embeddings that meet minimum similarity requirements.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pymilvus import Collection

from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import (
    ALL_TIER_COLLECTIONS,
    collection_for_tier,
    ensure_collection_loaded,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for a single search result with metadata."""

    iid: str
    tier: int
    credibility_score: float
    date: int
    domain: str
    similarity_score: float  # COSINE distance (higher = more similar)
    is_relevant: bool


class RelevanceFilter:
    """Filter search results based on relevance thresholds."""

    # Default thresholds for COSINE similarity (range: -1 to 1)
    # Higher values = stricter filtering
    DEFAULT_THRESHOLD = 0.5  # Moderate relevance
    STRICT_THRESHOLD = 0.7  # High relevance
    LENIENT_THRESHOLD = 0.3  # Low relevance

    @staticmethod
    def filter_by_threshold(
        results: List[SearchResult], threshold: float = DEFAULT_THRESHOLD
    ) -> tuple[List[SearchResult], List[SearchResult]]:
        """
        Split results into relevant and irrelevant based on threshold.

        Args:
            results: List of search results with similarity scores
            threshold: Minimum similarity score to be considered relevant

        Returns:
            Tuple of (relevant_results, irrelevant_results)
        """
        relevant = []
        irrelevant = []

        for result in results:
            if result.similarity_score >= threshold:
                result.is_relevant = True
                relevant.append(result)
            else:
                result.is_relevant = False
                irrelevant.append(result)

        logger.info(
            f"Filtered {len(results)} results: "
            f"{len(relevant)} relevant (≥{threshold}), "
            f"{len(irrelevant)} irrelevant (<{threshold})"
        )

        return relevant, irrelevant

    @staticmethod
    def get_adaptive_threshold(results: List[SearchResult]) -> float:
        """
        Calculate adaptive threshold based on score distribution.

        Uses mean + 0.5*stddev as threshold to filter out low-quality matches
        while keeping genuinely relevant results.
        """
        if not results:
            return RelevanceFilter.DEFAULT_THRESHOLD

        scores = [r.similarity_score for r in results]
        mean_score = sum(scores) / len(scores)

        # Calculate standard deviation
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance**0.5

        # Adaptive threshold: mean + 0.5*stddev
        adaptive_threshold = mean_score + (0.5 * std_dev)

        # Clamp between lenient and strict thresholds
        threshold = max(
            RelevanceFilter.LENIENT_THRESHOLD,
            min(adaptive_threshold, RelevanceFilter.STRICT_THRESHOLD),
        )

        logger.info(
            f"Adaptive threshold calculated: {threshold:.3f} "
            f"(mean={mean_score:.3f}, std={std_dev:.3f})"
        )

        result: float = float(threshold)
        return result


def search_with_relevance_filter(
    collection_name: Optional[str] = None,
    query: str = "",
    top_k: int = 10,
    relevance_threshold: Optional[float] = None,
    use_adaptive_threshold: bool = False,
    tier: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform vector search with automatic relevance filtering.

    Args:
        collection_name: Explicit Milvus collection to search (overrides tier).
        query: Search query text
        top_k: Number of results to retrieve before filtering
        relevance_threshold: Minimum similarity score (None = use default)
        use_adaptive_threshold: Calculate threshold from result distribution
        tier: Tier (1, 2, or 3) whose collection to search.
              When None and collection_name is None, all tiers are searched.

    Returns:
        Dictionary with relevant results, filtered results, and metadata
    """
    # Determine which collections to search
    collections_to_search: List[tuple[str, Optional[int]]]
    if collection_name is not None:
        collections_to_search = [(collection_name, None)]
    elif tier is not None:
        collections_to_search = [(collection_for_tier(tier), tier)]
    else:
        collections_to_search = [
            (name, idx + 1) for idx, name in enumerate(ALL_TIER_COLLECTIONS)
        ]

    display_name = collection_name or (
        collection_for_tier(tier) if tier else "all tiers"
    )
    logger.info(f"Searching '{display_name}' for query: '{query[:50]}...'")

    # Step 1: Generate query embedding
    query_embedding = embedding_generator.embed(query)

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    # Step 2: Search each target collection and merge results
    all_results: List[SearchResult] = []
    for col_name, inferred_tier in collections_to_search:
        ensure_collection_loaded(col_name)
        collection = Collection(col_name)

        search_results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=[
                "iid",
                "credibility_score",
                "date",
                "domain",
            ],
        )

        determined_tier = inferred_tier or 0
        for hit in search_results[0]:
            result = SearchResult(
                iid=str(hit.id),
                tier=determined_tier,
                credibility_score=float(getattr(hit, "credibility_score", 0.0)),
                date=int(getattr(hit, "date", 0)),
                domain=str(getattr(hit, "domain", "")),
                similarity_score=float(hit.distance),
                is_relevant=False,
            )
            all_results.append(result)

    # Sort merged results by similarity (descending) and keep top_k
    all_results.sort(key=lambda r: r.similarity_score, reverse=True)
    all_results = all_results[:top_k]

    # Step 4: Determine threshold
    if use_adaptive_threshold:
        threshold = RelevanceFilter.get_adaptive_threshold(all_results)
    elif relevance_threshold is not None:
        threshold = relevance_threshold
    else:
        threshold = RelevanceFilter.DEFAULT_THRESHOLD

    # Step 5: Filter by relevance
    relevant_results, irrelevant_results = RelevanceFilter.filter_by_threshold(
        all_results, threshold
    )

    # Step 6: Format response
    return {
        "query": query,
        "top_k": top_k,
        "threshold": threshold,
        "threshold_type": (
            "adaptive"
            if use_adaptive_threshold
            else "custom"
            if relevance_threshold is not None
            else "default"
        ),
        "total_results": len(all_results),
        "relevant_count": len(relevant_results),
        "irrelevant_count": len(irrelevant_results),
        "relevant_results": [
            {
                "iid": r.iid,
                "tier": r.tier,
                "credibility_score": r.credibility_score,
                "date": r.date,
                "domain": r.domain,
                "similarity_score": r.similarity_score,
            }
            for r in relevant_results
        ],
        "filtered_results": [
            {
                "iid": r.iid,
                "similarity_score": r.similarity_score,
                "reason": (
                    f"Below threshold " f"({r.similarity_score:.3f} < {threshold:.3f})"
                ),
            }
            for r in irrelevant_results
        ],
    }


# Convenience function for backward compatibility
def search_embeddings(
    collection_name: str,
    query: str,
    top_k: int = 10,
    min_relevance: float = RelevanceFilter.DEFAULT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Return only relevant results filtered by threshold.

    Returns:
        List of relevant results only (filtered by threshold)
    """
    response = search_with_relevance_filter(
        collection_name=collection_name,
        query=query,
        top_k=top_k,
        relevance_threshold=min_relevance,
    )

    result: list[dict[str, Any]] = response["relevant_results"]
    return result

"""Plain (legacy) vector search without relevance filtering.

Extracts the raw Milvus vector-similarity search that was previously inlined
in the ``POST /search`` API endpoint so it can be reused programmatically.
"""

import logging
from typing import Any, Dict, List, Optional

from pymilvus import Collection

from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import (
    ALL_TIER_COLLECTIONS,
    collection_for_tier,
    ensure_collection_loaded,
)

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    *,
    top_k: int = 10,
    collection_name: Optional[str] = None,
    tier: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run a vector similarity search against a Milvus tier collection.

    Parameters
    ----------
    query:
        Natural-language search query.
    top_k:
        Maximum number of nearest-neighbour results to return.
    collection_name:
        Explicit Milvus collection to search.  Overrides *tier*.
    tier:
        Tier (1, 2, or 3) whose collection to search.
        When ``None`` **and** *collection_name* is ``None``, all tier
        collections are searched and results merged.

    Returns
    -------
    List of hit dicts, each containing iid, tier, credibility_score, date,
    domain, and distance.
    """
    vec = embedding_generator.embed(query)
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

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

    hits: List[Dict[str, Any]] = []
    for col_name, inferred_tier in collections_to_search:
        ensure_collection_loaded(col_name)
        collection = Collection(col_name)

        results = collection.search(
            data=[vec],
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
        for hit in results[0]:
            hits.append(
                {
                    "iid": str(hit.id),
                    "tier": determined_tier,
                    "credibility_score": float(getattr(hit, "credibility_score", 0.0)),
                    "date": int(getattr(hit, "date", 0)),
                    "domain": str(getattr(hit, "domain", "")),
                    "distance": float(hit.distance),
                }
            )

    # Sort merged results by distance (higher = more similar for COSINE)
    hits.sort(key=lambda h: h["distance"], reverse=True)
    return hits[:top_k]

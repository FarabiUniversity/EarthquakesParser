"""Plain (legacy) vector search without relevance filtering.

Extracts the raw Milvus vector-similarity search that was previously inlined
in the ``POST /search`` API endpoint so it can be reused programmatically.
"""

import logging
from typing import Any, Dict, List, Optional

from pymilvus import Collection

from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import COLLECTION_NAME, ensure_collection_loaded

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    *,
    top_k: int = 10,
    collection_name: str = COLLECTION_NAME,
    tier: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run a vector similarity search against the Milvus collection.

    Parameters
    ----------
    query:
        Natural-language search query.
    top_k:
        Maximum number of nearest-neighbour results to return.
    collection_name:
        Milvus collection to search.
    tier:
        Optional tier filter (1, 2, or 3).  When ``None`` all tiers are searched.

    Returns
    -------
    List of hit dicts, each containing iid, tier, credibility_score, date,
    domain, and distance.
    """
    ensure_collection_loaded(collection_name)
    vec = embedding_generator.embed(query)
    collection = Collection(collection_name)
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    expr = f"tier == {tier}" if tier is not None else None

    results = collection.search(
        data=[vec],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "iid",
            "tier",
            "credibility_score",
            "date",
            "domain",
        ],
    )

    hits: List[Dict[str, Any]] = []
    for hit in results[0]:
        hits.append(
            {
                "iid": str(hit.id),
                "tier": int(getattr(hit, "tier", 0)),
                "credibility_score": float(getattr(hit, "credibility_score", 0.0)),
                "date": int(getattr(hit, "date", 0)),
                "domain": str(getattr(hit, "domain", "")),
                "distance": float(hit.distance),
            }
        )

    return hits

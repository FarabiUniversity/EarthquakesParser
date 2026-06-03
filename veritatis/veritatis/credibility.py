"""Credibility score computation based on vector consensus.

This module combines SimilarityDetector and vector_consensus to calculate
credibility scores for all records in Tier 1. Records that are similar to
many other records (consensus) receive higher credibility scores.
"""

import logging
from typing import Any, Dict, List, Optional

from pymilvus import Collection

from veritatis.ingestion import get_store
from veritatis.similarity import SimilarityDetector
from veritatis.vector_consensus import (
    _fetch_supabase_main_text_by_iids,
    find_best_vector_with_embeddings,
)
from veritatis.vector_stores import ensure_collection_loaded

logger = logging.getLogger(__name__)


def _calibrate_group_scores(
    ranked_analyses: List[Any], group_size: int, neutral_score: float
) -> Dict[str, float]:
    """Map per-group ranked scores into a conservative credibility range.

    Raw combined scores from consensus can saturate near 1.0 in small groups.
    This calibration keeps rankings but scales certainty by group size.
    """
    if not ranked_analyses:
        return {}

    # Confidence grows with group size; cap at 1.0 for >=5 records.
    confidence = min(1.0, max(0.0, (group_size - 1) / 4.0))
    cap = min(0.95, neutral_score + 0.45 * confidence)
    floor = max(0.0, neutral_score - 0.20 * (1.0 - confidence))
    floor = min(floor, neutral_score)

    raw_scores = [float(a.combined_score) for a in ranked_analyses]
    min_raw = min(raw_scores)
    max_raw = max(raw_scores)

    if max_raw == min_raw:
        midpoint = max(neutral_score, (floor + cap) / 2.0)
        return {a.iid: midpoint for a in ranked_analyses}

    out: Dict[str, float] = {}
    span = cap - floor
    for analysis in ranked_analyses:
        raw = float(analysis.combined_score)
        normalized = (raw - min_raw) / (max_raw - min_raw)
        out[analysis.iid] = floor + normalized * span
    return out


def _get_record_embedding(collection, iid: str) -> Optional[List[float]]:
    """Fetch embedding for a specific record."""
    try:
        field_names = [f.name for f in collection.schema.fields]
        results = collection.query(
            expr=f'iid == "{iid}"',
            output_fields=field_names,
        )
        if results and len(results) > 0:
            embedding = results[0].get("embedding")
            if embedding is not None:
                return list(embedding)
        return None
    except Exception as e:
        logger.error(f"Error retrieving embedding for {iid}: {e}")
        return None


def compute_credibility_scores(
    collection_name: str = "veritatis_tier1_lake",
    similarity_threshold: float = 0.85,
    min_group_size: int = 2,
    centrality_weight: float = 0.6,
    detail_weight: float = 0.4,
    neutral_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute and update credibility scores for all records in Tier 1.

    Algorithm:
    1. Use SimilarityDetector to find groups of similar records
    2. For each group, use find_best_vector_with_embeddings to rank records
    3. Update credibility_score in Milvus for each record in the group
    4. Assign neutral_score (default 0.5) to records not in any group

    Args:
        collection_name: Milvus collection name (default: veritatis_tier1_lake)
        similarity_threshold: Minimum cosine similarity to group records (0-1)
        min_group_size: Minimum number of records to form a group
        centrality_weight: Weight for centrality score in consensus ranking
        detail_weight: Weight for detail score (text length/quality)
        neutral_score: Score for records without similar neighbors

    Returns:
        Dictionary with computation statistics:
        - total_records: Total number of records processed
        - groups_found: Number of similarity groups found
        - records_in_groups: Number of records in groups
        - records_without_groups: Number of singleton records
        - updated_count: Total number of records updated
    """
    logger.info(
        f"Starting credibility score computation for '{collection_name}' "
        f"(similarity_threshold={similarity_threshold}, "
        f"min_group_size={min_group_size})"
    )

    # Step 1: Find similarity groups
    detector = SimilarityDetector(
        collection_name=collection_name,
        similarity_threshold=similarity_threshold,
        min_group_size=min_group_size,
    )
    groups = detector.find_similar_groups()

    logger.info(f"Found {len(groups)} similarity groups")

    if not groups:
        logger.warning(
            "No similarity groups found - all records will get neutral score"
        )

    store = get_store()
    ensure_collection_loaded(collection_name)
    collection = Collection(collection_name)

    # Track which records were processed
    processed_iids = set()
    updates: List[Dict[str, Any]] = []

    # Step 3: Process each group
    for i, group in enumerate(groups):
        logger.info(
            f"Processing group {i+1}/{len(groups)}: "
            f"anchor={group.anchor_iid[:8]}..., size={group.group_size}"
        )

        # Get all iids in this group
        all_iids = group.get_all_iids()

        # Fetch embeddings for all records in the group
        vectors_with_embeddings = []
        for iid in all_iids:
            embedding = _get_record_embedding(collection, iid)
            if embedding is None:
                logger.warning(f"Could not get embedding for {iid}, skipping")
                continue

            # Get metadata from Milvus
            metadata_results = collection.query(
                expr=f'iid == "{iid}"',
                output_fields=["iid", "credibility_score", "date", "domain"],
            )
            if not metadata_results:
                logger.warning(f"Could not get metadata for {iid}, skipping")
                continue

            metadata = metadata_results[0]

            vectors_with_embeddings.append(
                {
                    "iid": iid,
                    "main_text": "",  # Will be filled by vector_consensus
                    "credibility_score": float(metadata.get("credibility_score", 0.0)),
                    "date": int(metadata.get("date", 0)),
                    "domain": str(metadata.get("domain", "")),
                    "embedding": embedding,
                }
            )

        if len(vectors_with_embeddings) < min_group_size:
            logger.warning(
                f"Group has insufficient embeddings "
                f"({len(vectors_with_embeddings)}), skipping"
            )
            continue

        # Fetch main_text from Supabase
        iids_for_content: List[str] = [str(v["iid"]) for v in vectors_with_embeddings]
        try:
            content_map = _fetch_supabase_main_text_by_iids(iids_for_content)
        except Exception as e:
            logger.warning(
                "Failed to fetch Supabase main_text; continuing with empty text for "
                f"group {i+1}: {e}"
            )
            content_map = {}
        for v in vectors_with_embeddings:
            iid_str: str = str(v["iid"])
            v["main_text"] = content_map.get(iid_str, "")

        # Rank vectors by consensus
        try:
            _, all_ranked = find_best_vector_with_embeddings(
                vectors_with_embeddings,
                centrality_weight=centrality_weight,
                detail_weight=detail_weight,
            )
        except Exception as e:
            logger.error(f"Error ranking vectors in group {i+1}: {e}")
            continue

        calibrated_scores = _calibrate_group_scores(
            all_ranked, group.group_size, neutral_score
        )

        # Prepare updates for this group
        for analysis in all_ranked:
            updates.append(
                {
                    "iid": analysis.iid,
                    "credibility_score": float(
                        calibrated_scores.get(analysis.iid, neutral_score)
                    ),
                }
            )
            processed_iids.add(analysis.iid)

        logger.info(
            f"Group {i+1} processed: {len(all_ranked)} records ranked "
            f"(best score: {all_ranked[0].combined_score:.3f})"
        )

    # Step 4: Handle records without groups (neutral score)
    all_records = collection.query(
        expr="iid != ''",
        output_fields=["iid"],
        limit=10000,
    )
    all_iids_in_collection = {r["iid"] for r in all_records}
    unprocessed_iids = all_iids_in_collection - processed_iids

    logger.info(
        f"Found {len(unprocessed_iids)} records without groups "
        f"(assigning neutral score {neutral_score})"
    )

    for iid in unprocessed_iids:
        updates.append(
            {
                "iid": iid,
                "credibility_score": neutral_score,
            }
        )

    # Step 5: Update all credibility scores in Milvus
    if updates:
        logger.info(f"Updating credibility scores for {len(updates)} records...")
        updated_count = store.update_credibility_scores(collection_name, updates)
        logger.info(f"Successfully updated {updated_count} records")
    else:
        updated_count = 0
        logger.warning("No updates to apply")

    # Return statistics
    stats = {
        "total_records": len(all_iids_in_collection),
        "groups_found": len(groups),
        "records_in_groups": len(processed_iids),
        "records_without_groups": len(unprocessed_iids),
        "updated_count": updated_count,
        "parameters": {
            "similarity_threshold": similarity_threshold,
            "min_group_size": min_group_size,
            "centrality_weight": centrality_weight,
            "detail_weight": detail_weight,
            "neutral_score": neutral_score,
        },
    }

    logger.info(
        f"Credibility computation complete: "
        f"{stats['total_records']} total, "
        f"{stats['records_in_groups']} in groups, "
        f"{stats['records_without_groups']} without groups"
    )

    return stats

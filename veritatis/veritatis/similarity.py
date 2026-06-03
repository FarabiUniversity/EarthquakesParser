"""
Similarity detection module for finding semantically similar embeddings in Tier 1.

This module uses Milvus's built-in vector search to find groups of similar content
that should be consolidated and moved to Tier 2 after summarization.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from pymilvus import Collection

from veritatis.ingest_parsed_content import _get_supabase_client
from veritatis.vector_stores import ensure_collection_loaded

logger = logging.getLogger(__name__)

# Supabase client setup (reused from vector_consensus.py pattern)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

_SUPABASE_CLIENT: Any = None


def _fetch_main_text_by_iids(iids: List[str]) -> Dict[str, str]:
    """Fetch main_text from Supabase for given iids."""
    if not iids:
        return {}

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL/SUPABASE_KEY not set; returning empty content")
        return {}

    client = _get_supabase_client()
    out: Dict[str, str] = {}

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
            logger.error(f"Failed to fetch parsed_content.main_text: {e}")
            continue

        rows = getattr(resp, "data", None) or []
        for row in rows:
            try:
                rid = str(row.get("id", ""))
            except (AttributeError, TypeError):
                continue
            if not rid:
                continue
            main_text = row.get("main_text", "")
            if isinstance(main_text, str):
                out[rid] = main_text
            else:
                out[rid] = str(main_text) if main_text else ""
    return out


@dataclass
class SimilarityGroup:
    """Represents a group of similar embeddings."""

    anchor_iid: str  # The record iid used as the search anchor
    anchor_content: str  # Fetched from Supabase
    similar_records: List[Dict[str, Any]]  # List of similar records with metadata
    similarity_threshold: float
    group_size: int

    def get_all_iids(self) -> List[str]:
        """Return all record iids in this group (anchor + similar records)."""
        iids = [self.anchor_iid]
        iids.extend([r["iid"] for r in self.similar_records])
        return iids

    def get_all_contents(self) -> List[str]:
        """Return all content texts for summarization."""
        contents = [self.anchor_content]
        contents.extend([r["content"] for r in self.similar_records])
        return contents

    # Backwards compatibility alias
    @property
    def anchor_id(self) -> str:
        """Alias for anchor_iid (backwards compatibility)."""
        return self.anchor_iid

    def get_all_ids(self) -> List[str]:
        """Alias for get_all_iids (backwards compatibility)."""
        return self.get_all_iids()


class SimilarityDetector:
    """
    Detect groups of similar embeddings using Milvus vector search.

    Strategy:
    1. Iterate through all records in Tier 1
    2. For each record, search for similar records (cosine similarity above threshold)
    3. Group similar records together
    4. Return groups that can be summarized and moved to Tier 2
    """

    # Default similarity threshold (COSINE similarity 0-1, higher = more similar)
    DEFAULT_SIMILARITY_THRESHOLD = 0.85  # Very high similarity
    MIN_GROUP_SIZE = 2  # At least 2 records needed to form a group

    def __init__(
        self,
        collection_name: str = "veritatis_tier1_lake",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_group_size: int = MIN_GROUP_SIZE,
        max_search_results: int = 20,
    ):
        """
        Initialize similarity detector.

        Args:
            collection_name: Milvus collection to search
            similarity_threshold: Minimum cosine similarity (0-1) to consider
                records similar
            min_group_size: Minimum number of records in a group
            max_search_results: Maximum number of similar records to retrieve
                per anchor
        """
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.min_group_size = min_group_size
        self.max_search_results = max_search_results
        self.processed_ids: Set[str] = set()  # Track already processed records

    def find_similar_groups(
        self,
        limit: Optional[int] = None,
        skip_processed: bool = True,
    ) -> List[SimilarityGroup]:
        """
        Find all groups of similar records in Tier 1.

        Args:
            limit: Maximum number of groups to return (None = unlimited)
            skip_processed: Skip records already in processed_ids set

        Returns:
            List of SimilarityGroup objects with content fetched from Supabase
        """
        logger.info(
            f"Starting similarity detection in '{self.collection_name}' "
            f"(threshold={self.similarity_threshold}, "
            f"min_group_size={self.min_group_size})"
        )

        ensure_collection_loaded(self.collection_name)
        collection = Collection(self.collection_name)

        # Step 1: Get all records from Tier 1
        all_records = self._get_all_records(collection)
        logger.info(f"Found {len(all_records)} total records in {self.collection_name}")

        if not all_records:
            logger.warning("No records found in collection")
            return []

        # Step 2: Fetch content from Supabase for all records
        all_iids = [r["iid"] for r in all_records]
        content_map = _fetch_main_text_by_iids(all_iids)
        logger.info(f"Fetched content for {len(content_map)} records from Supabase")

        # Step 3: For each record, find similar neighbors
        similarity_groups = []
        for record in all_records:
            record_iid = record["iid"]

            # Skip if already processed (part of another group)
            if skip_processed and record_iid in self.processed_ids:
                continue

            # Search for similar records using this record's embedding
            similar_records = self._find_similar_to_record(collection, record)

            # Filter out already processed records
            if skip_processed:
                similar_records = [
                    r for r in similar_records if r["iid"] not in self.processed_ids
                ]

            # Check if we have enough similar records to form a group
            if len(similar_records) >= (
                self.min_group_size - 1
            ):  # -1 because anchor counts
                # Fill in content for similar records
                for r in similar_records:
                    r["content"] = content_map.get(r["iid"], "")

                group = SimilarityGroup(
                    anchor_iid=record_iid,
                    anchor_content=content_map.get(record_iid, ""),
                    similar_records=similar_records,
                    similarity_threshold=self.similarity_threshold,
                    group_size=len(similar_records) + 1,
                )

                similarity_groups.append(group)

                # Mark all records in this group as processed
                for iid_in_group in group.get_all_iids():
                    self.processed_ids.add(iid_in_group)

                logger.info(
                    f"Found similarity group: "
                    f"anchor={record_iid[:8]}..., "
                    f"size={group.group_size}"
                )

                # Check limit
                if limit and len(similarity_groups) >= limit:
                    break

        logger.info(
            f"Similarity detection complete: "
            f"found {len(similarity_groups)} groups "
            f"({len(self.processed_ids)} total records processed)"
        )

        return similarity_groups

    def _get_all_records(self, collection: Collection) -> List[Dict[str, Any]]:
        """
        Retrieve all records from the collection.

        Returns list of dicts with iid and metadata. Content is NOT included here
        (must be fetched from Supabase separately).
        """
        try:
            # Query all records - use iid instead of id
            results = collection.query(
                expr="iid != ''",  # Match all records with non-empty iid
                output_fields=["iid", "credibility_score", "date", "domain"],
                limit=10000,  # Adjust based on expected Tier 1 size
            )
            return list(results)
        except Exception as e:
            logger.error(f"Error retrieving records: {e}")
            return []

    def _find_similar_to_record(
        self, collection: Collection, anchor_record: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find records similar to the given anchor record using vector search.

        Args:
            collection: Milvus collection
            anchor_record: The record to find similar neighbors for

        Returns:
            List of similar records (excluding the anchor itself)
        """
        anchor_iid = anchor_record["iid"]

        # Get the embedding for this record
        try:
            anchor_embedding = self._get_record_embedding(collection, anchor_iid)
            if anchor_embedding is None:
                logger.warning(f"Could not retrieve embedding for {anchor_iid}")
                return []
        except Exception as e:
            logger.error(f"Error getting embedding for {anchor_iid}: {e}")
            return []

        # Perform vector search to find similar records
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        try:
            search_results = collection.search(
                data=[anchor_embedding],
                anns_field="embedding",
                param=search_params,
                limit=self.max_search_results
                + 1,  # +1 because anchor will be in results
                output_fields=["iid", "credibility_score", "date", "domain"],
            )
        except Exception as e:
            logger.error(f"Error searching for similar records: {e}")
            return []

        # Process results
        similar_records = []
        for hit in search_results[0]:
            hit_iid = str(hit.entity.get("iid"))
            similarity_score = float(hit.distance)

            # Skip the anchor itself
            if hit_iid == anchor_iid:
                continue

            # Only include if similarity is above threshold
            if similarity_score >= self.similarity_threshold:
                similar_records.append(
                    {
                        "iid": hit_iid,
                        "credibility_score": float(
                            hit.entity.get("credibility_score", 0.0)
                        ),
                        "date": int(hit.entity.get("date", 0)),
                        "domain": str(hit.entity.get("domain", "")),
                        "similarity_score": similarity_score,
                        "content": "",  # Will be filled later from Supabase
                    }
                )

        return similar_records

    def _get_record_embedding(
        self, collection: Collection, record_iid: str
    ) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a specific record.

        Milvus doesn't return vector fields in query() by default,
        so we need to use get() method or search with the iid.
        """
        try:
            # Get all field names including embedding
            field_names = [f.name for f in collection.schema.fields]

            # Query to get the full record including embedding
            results = collection.query(
                expr=f'iid == "{record_iid}"',
                output_fields=field_names,  # Include all fields
            )

            if results and len(results) > 0:
                embedding = results[0].get("embedding")
                if embedding is not None:
                    return list(embedding)
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for {record_iid}: {e}")
            return None

    def process_similarity_groups_batch(
        self,
        groups: List[SimilarityGroup],
        summarization_callback: Callable[[List[str]], Dict[str, Any]],
        tier2_insertion_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process similarity groups: summarize and optionally insert to Tier 2.

        Args:
            groups: List of similarity groups to process
            summarization_callback: Function that takes list of contents
                and returns summary dict. Expected signature:
                fn(contents: List[str]) -> Dict[str, Any]
                Return dict should contain: {
                    "summary": str,
                    "confidence": float,
                    etc.
                }
            tier2_insertion_callback: Optional function to insert
                summarized record to Tier 2.
                Signature: fn(record: Dict[str, Any]) -> None

        Returns:
            List of processed results with summaries and metadata
        """
        logger.info(f"Processing {len(groups)} similarity groups for summarization")
        results = []

        for i, group in enumerate(groups):
            try:
                logger.info(
                    f"Processing group {i+1}/{len(groups)}: "
                    f"anchor={group.anchor_id[:8]}..., size={group.group_size}"
                )

                # Call the summarization function (provided by your colleague)
                contents = group.get_all_contents()
                summary_result = summarization_callback(contents)

                # Prepare result
                processed_result = {
                    "group_id": f"group_{i}",
                    "anchor_id": group.anchor_id,
                    "record_ids": group.get_all_ids(),
                    "group_size": group.group_size,
                    "summary": summary_result,
                    "status": "summarized",
                }

                # Optionally insert to Tier 2
                if tier2_insertion_callback:
                    try:
                        tier2_insertion_callback(processed_result)
                        processed_result["status"] = "moved_to_tier2"
                    except Exception as e:
                        logger.error(f"Error inserting to Tier 2: {e}")
                        processed_result["status"] = "summarized_but_not_moved"
                        processed_result["error"] = str(e)

                results.append(processed_result)

            except Exception as e:
                logger.error(f"Error processing group {i}: {e}")
                results.append(
                    {
                        "group_id": f"group_{i}",
                        "anchor_id": group.anchor_id,
                        "error": str(e),
                        "status": "failed",
                    }
                )

        moved_count = sum(1 for r in results if r["status"] == "moved_to_tier2")
        logger.info(
            f"Batch processing complete: {len(results)} groups processed, "
            f"{moved_count} moved to Tier 2"
        )

        return results

    def reset_processed(self):
        """Clear the processed IDs set to allow reprocessing."""
        self.processed_ids.clear()
        logger.info("Reset processed IDs")

#!/usr/bin/env python3
"""
Compute credibility scores for Tier 1 records.

This script computes consensus-based credibility scores for all records
in the Tier 1 Milvus collection by finding groups of similar articles
and ranking them by centrality.

Usage:
    python scripts/compute_credibility.py
    python scripts/compute_credibility.py --threshold 0.90 --min-group-size 3
    python scripts/compute_credibility.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from veritatis.credibility import compute_credibility_scores

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Execute the credibility score computation."""
    parser = argparse.ArgumentParser(
        description="Compute credibility scores for Tier 1 records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default parameters
  python scripts/compute_credibility.py

  # High similarity threshold (stricter grouping)
  python scripts/compute_credibility.py --threshold 0.90

  # Require larger groups
  python scripts/compute_credibility.py --min-group-size 5

  # Adjust consensus/detail balance
  python scripts/compute_credibility.py --centrality-weight 0.7 --detail-weight 0.3

  # Different collection
  python scripts/compute_credibility.py --collection veritatis_tier2_arena
        """,
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="veritatis_tier1_lake",
        help="Milvus collection number (default: veritatis_tier1_lake)",
    )

    parser.add_argument(
        "--threshold",
        "--similarity-threshold",
        type=float,
        default=0.85,
        dest="similarity_threshold",
        help="Minimum cosine similarity (0-1) to group records (default: 0.85)",
    )

    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum number of records to form a group (default: 2)",
    )

    parser.add_argument(
        "--centrality-weight",
        type=float,
        default=0.6,
        help="Weight for centrality score (default: 0.6)",
    )

    parser.add_argument(
        "--detail-weight",
        type=float,
        default=0.4,
        help="Weight for detail score (default: 0.4)",
    )

    parser.add_argument(
        "--neutral-score",
        type=float,
        default=0.5,
        help="Score for records without groups (default: 0.5)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger("veritatis").setLevel(logging.DEBUG)

    # Validate weights
    total_weight = args.centrality_weight + args.detail_weight
    if abs(total_weight - 1.0) > 1e-6:
        logger.error(
            f"Weights must sum to 1.0, got {total_weight:.6f} "
            f"(centrality={args.centrality_weight}, detail={args.detail_weight})"
        )
        sys.exit(1)

    # Print configuration
    print("=" * 70)
    print("Credibility Score Computation")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Collection:          {args.collection}")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Min group size:      {args.min_group_size}")
    print(f"  Centrality weight:   {args.centrality_weight}")
    print(f"  Detail weight:       {args.detail_weight}")
    print(f"  Neutral score:       {args.neutral_score}")
    print()

    try:
        # Compute credibility scores
        stats = compute_credibility_scores(
            collection_name=args.collection,
            similarity_threshold=args.similarity_threshold,
            min_group_size=args.min_group_size,
            centrality_weight=args.centrality_weight,
            detail_weight=args.detail_weight,
            neutral_score=args.neutral_score,
        )

        # Print results
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print("\n✓ Computation complete!")
        print("\nStatistics:")
        print(f"  Total records:         {stats['total_records']}")
        print(f"  Groups found:          {stats['groups_found']}")
        print(f"  Records in groups:     {stats['records_in_groups']}")
        print(f"  Records without groups: {stats['records_without_groups']}")
        print(f"  Updated count:         {stats['updated_count']}")
        print()

        if stats["groups_found"] > 0:
            avg_group_size = stats["records_in_groups"] / stats["groups_found"]
            print(f"  Average group size:    {avg_group_size:.1f} records")
            coverage = (stats["records_in_groups"] / stats["total_records"]) * 100
            print(f"  Group coverage:        {coverage:.1f}%")
        print()

        # Summary message
        if stats["records_in_groups"] == 0:
            print("⚠️  No similarity groups found. Consider:")
            print("   - Lowering --threshold (e.g., 0.80)")
            print("   - Reducing --min-group-size (e.g., 2)")
            print("   - Checking if records have embeddings")
        else:
            print(f"✓ Successfully updated {stats['updated_count']} credibility scores")

        print("\n" + "=" * 70)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error computing credibility scores: {e}", exc_info=True)
        print("\n" + "=" * 70)
        print(f"✗ Error: {e}")
        print("\nCheck logs above for details.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()

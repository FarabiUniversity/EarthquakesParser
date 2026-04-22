# Similarity Detection in Veritatis

This document explains the similarity detection feature for finding and consolidating similar earthquake-related content from Tier 1.

## Overview

The similarity detection system uses **Milvus's built-in vector search** to find semantically similar embeddings without requiring any external clustering algorithms. This is efficient because Milvus is already optimized for vector similarity search.

## How It Works

### 1. Detection Strategy

```
For each record in Tier 1:
  1. Use the record as an "anchor"
  2. Search Milvus for similar records (using the anchor's embedding)
  3. Filter results by similarity threshold (default: 0.85 cosine similarity)
  4. Group similar records together
  5. Skip records already processed (avoid duplicates)
```

### 2. Key Components

#### `SimilarityDetector` Class
Located in `veritatis/similarity.py`

**Main Methods:**
- `find_similar_groups()` - Detects all similarity groups in Tier 1
- `process_similarity_groups_batch()` - Processes groups with summarization callback
- `_find_similar_to_record()` - Uses Milvus vector search for similarity

**Configuration:**
- `similarity_threshold`: Minimum cosine similarity (0-1), default 0.85
- `min_group_size`: Minimum records per group, default 2
- `max_search_results`: Maximum similar records per anchor, default 20

## API Endpoints

### 1. Detect Similar Groups (No Processing)

**Endpoint:** `POST /similarity/detect`

**Purpose:** Find groups of similar records without summarization

**Request Body:**
```json
{
  "similarity_threshold": 0.85,
  "min_group_size": 2,
  "max_groups": 10,
  "collection": "veritatis_tier1_lake"
}
```

**Response:**
```json
{
  "collection": "veritatis_tier1_lake",
  "similarity_threshold": 0.85,
  "min_group_size": 2,
  "groups_found": 5,
  "total_records_in_groups": 23,
  "groups": [
    {
      "anchor_id": "abc123...",
      "anchor_content": "Magnitude 6.2 earthquake hits Turkey...",
      "group_size": 4,
      "similar_record_ids": ["def456...", "ghi789...", "jkl012..."],
      "similarity_scores": [0.92, 0.89, 0.87]
    }
  ]
}
```

**Usage:**
```bash
curl -X POST http://localhost:8000/similarity/detect \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.85,
    "min_group_size": 2,
    "max_groups": 10
  }'
```

### 2. Process Similar Groups (With Summarization)

**Endpoint:** `POST /similarity/process`

**Purpose:** Find groups AND call summarization function for each group

**Request Body:**
```json
{
  "similarity_threshold": 0.85,
  "min_group_size": 2,
  "max_groups": 10,
  "collection": "veritatis_tier1_lake"
}
```

**Response:**
```json
{
  "status": "processed",
  "groups_found": 5,
  "groups_processed": 5,
  "results": [
    {
      "group_id": "group_0",
      "anchor_id": "abc123...",
      "record_ids": ["abc123...", "def456...", "ghi789..."],
      "group_size": 3,
      "summary": {
        "summary": "Multiple sources report magnitude 6.2 earthquake in Turkey...",
        "confidence": 0.92
      },
      "status": "summarized"
    }
  ]
}
```

**Usage:**
```bash
curl -X POST http://localhost:8000/similarity/process \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.85,
    "min_group_size": 2,
    "max_groups": 10
  }'
```

## Integration with Your Colleague's Summarization Function

Your colleague needs to implement a function that takes a list of content strings and returns a summary dictionary.

### Function Signature

```python
def summarize_similar_content(contents: List[str]) -> Dict[str, Any]:
    """
    Summarize a group of similar earthquake-related content.

    Args:
        contents: List of text content from similar records

    Returns:
        Dictionary with:
        - "summary": str - The consolidated summary
        - "confidence": float - Confidence score (0-1)
        - Any other metadata your colleague wants to include
    """
    # Your colleague's implementation here
    # Could use LLM, extractive summarization, etc.

    summary = "..."  # Generate summary from contents
    confidence = 0.95  # Calculate confidence

    return {
        "summary": summary,
        "confidence": confidence,
        "source_count": len(contents),
        "method": "llm_summarization"
    }
```

### Integration Steps

1. **Replace the placeholder in `api/main.py`:**

Current code (line ~400):
```python
def placeholder_summarization(contents: list[str]) -> dict[str, Any]:
    """Placeholder - your colleague will implement the real summarization."""
    return {
        "summary": f"Summary of {len(contents)} similar records (placeholder)",
        "confidence": 0.0,
        "note": "Replace this with real LLM summarization",
    }
```

Replace with:
```python
from your_module import summarize_similar_content  # Import colleague's function

# In the endpoint:
results = detector.process_similarity_groups_batch(
    groups=groups,
    summarization_callback=summarize_similar_content,  # Use real function
    tier2_insertion_callback=None,  # Add if needed
)
```

2. **Optionally add Tier 2 insertion:**

If you want to automatically move summarized content to Tier 2:

```python
def insert_to_tier2(processed_result: Dict[str, Any]) -> None:
    """Insert summarized content to Tier 2."""
    summary_data = processed_result["summary"]

    # Generate embedding for the summary
    summary_embedding = embedding_generator.embed(summary_data["summary"])

    # Create Tier 2 record
    tier2_record = {
        "id": processed_result["anchor_id"],  # Or generate new ID
        "content": summary_data["summary"],
        "embedding": summary_embedding,
        "verification_confidence": summary_data["confidence"],
        "cross_source_count": processed_result["group_size"],
        "supabase_id": "",  # Fill if needed
    }

    # Insert to Tier 2
    _store.insert_record("veritatis_tier2_arena", tier2_record)

# Use it:
results = detector.process_similarity_groups_batch(
    groups=groups,
    summarization_callback=summarize_similar_content,
    tier2_insertion_callback=insert_to_tier2,
)
```

## Example Workflow

### Step 1: Detect Similar Groups
```python
# In your code or via API
detector = SimilarityDetector(
    collection_name="veritatis_tier1_lake",
    similarity_threshold=0.85,
    min_group_size=2,
)

groups = detector.find_similar_groups(limit=10)
print(f"Found {len(groups)} similarity groups")

for group in groups:
    print(f"Group: {group.anchor_id}")
    print(f"  Size: {group.group_size}")
    print(f"  Contents: {group.get_all_contents()}")
```

### Step 2: Process with Summarization
```python
# Define or import your summarization function
def my_summarization(contents: List[str]) -> Dict[str, Any]:
    # Use LLM or other method
    combined = " ".join(contents)
    summary = generate_summary(combined)  # Your logic
    return {
        "summary": summary,
        "confidence": 0.9,
    }

# Process groups
results = detector.process_similarity_groups_batch(
    groups=groups,
    summarization_callback=my_summarization,
)

for result in results:
    print(f"Processed: {result['anchor_id']}")
    print(f"  Summary: {result['summary']['summary']}")
    print(f"  Status: {result['status']}")
```

## Why This Approach Works

### 1. **Leverages Milvus Native Capabilities**
- No external clustering algorithms needed (K-means, DBSCAN, etc.)
- Milvus vector search is already optimized for similarity
- Uses HNSW index for fast approximate nearest neighbor search

### 2. **Efficient Processing**
- Query all records once
- For each record, use vector search (O(log n) with HNSW)
- No need to compute pairwise distances

### 3. **Flexible Thresholds**
- Adjust `similarity_threshold` based on your needs:
  - 0.95+: Nearly identical content
  - 0.85-0.95: Highly similar content (default)
  - 0.70-0.85: Moderately similar content
  - <0.70: Loosely related content

### 4. **Avoids Double Processing**
- Tracks `processed_ids` to skip records already in groups
- Each record appears in at most one group

## Configuration Recommendations

### For High Precision (Few False Positives)
```python
detector = SimilarityDetector(
    similarity_threshold=0.90,  # High threshold
    min_group_size=3,           # Larger groups
    max_search_results=10,      # Fewer candidates
)
```

### For High Recall (Catch More Similarities)
```python
detector = SimilarityDetector(
    similarity_threshold=0.75,  # Lower threshold
    min_group_size=2,           # Smaller groups
    max_search_results=50,      # More candidates
)
```

### For Balanced Approach (Recommended)
```python
detector = SimilarityDetector(
    similarity_threshold=0.85,  # Default
    min_group_size=2,           # Default
    max_search_results=20,      # Default
)
```

## Monitoring and Debugging

### Check Group Quality
```python
groups = detector.find_similar_groups()

for group in groups:
    print(f"\nGroup {group.anchor_id[:8]}:")
    print(f"  Anchor: {group.anchor_content[:100]}")

    for record in group.similar_records:
        print(f"  - [{record['similarity_score']:.3f}] {record['content'][:100]}")
```

### Reset Processed Records
If you want to reprocess everything:
```python
detector.reset_processed()
groups = detector.find_similar_groups()  # Will find all groups again
```

## Future Enhancements

1. **Incremental Processing**
   - Process only new records since last run
   - Store processed IDs in database

2. **Advanced Filtering**
   - Filter by time window (recent earthquakes)
   - Filter by location (embeddings from specific regions)
   - Filter by credibility score

3. **Hierarchical Clustering**
   - Find clusters of clusters
   - Multi-level summarization

4. **Quality Metrics**
   - Calculate inter-group similarity
   - Validate group coherence
   - Track summarization quality

## Troubleshooting

### Issue: No groups found
**Possible causes:**
- Similarity threshold too high (try lowering to 0.75)
- Not enough records in Tier 1
- Records are all dissimilar

### Issue: Too many groups
**Possible causes:**
- Similarity threshold too low (try raising to 0.90)
- Many duplicate/near-duplicate records

### Issue: Groups too large
**Possible causes:**
- Threshold too low
- Generic content matching everything

### Issue: Performance slow
**Solutions:**
- Reduce `max_search_results`
- Process in smaller batches (use `limit` parameter)
- Ensure Milvus indexes are built

## Questions for Your Colleague

When integrating the summarization function, discuss:

1. **Input format**: Is list of strings sufficient, or need metadata?
2. **Output format**: What fields should summary dict contain?
3. **Async vs Sync**: Should summarization be async (if using LLM API)?
4. **Error handling**: How to handle summarization failures?
5. **Batch size**: Process groups one-by-one or in batches?
6. **Tier 2 insertion**: Should this endpoint handle it, or separate step?

## Example Integration

Here's a complete example of integrating with your colleague's function:

```python
# In api/main.py or separate module

from your_colleague_module import generate_earthquake_summary
from veritatis.embeddings import embedding_generator
from veritatis.vector_stores import MilvusRecordStore

async def process_similarity_groups_endpoint():
    """Complete integration example."""

    # Initialize
    detector = SimilarityDetector(
        collection_name="veritatis_tier1_lake",
        similarity_threshold=0.85,
        min_group_size=2,
    )

    # Find groups
    groups = detector.find_similar_groups(limit=10)

    # Define summarization callback
    def summarize_earthquake_group(contents: List[str]) -> Dict[str, Any]:
        """Wrapper around colleague's function."""
        summary_text = generate_earthquake_summary(contents)

        return {
            "summary": summary_text,
            "confidence": 0.9,  # Or calculate from colleague's output
            "source_count": len(contents),
        }

    # Define Tier 2 insertion callback
    def insert_to_tier2(processed: Dict[str, Any]) -> None:
        """Insert summarized content to Tier 2."""
        summary_embedding = embedding_generator.embed(
            processed["summary"]["summary"]
        )

        tier2_record = {
            "id": processed["anchor_id"],
            "content": processed["summary"]["summary"],
            "embedding": summary_embedding,
            "verification_confidence": processed["summary"]["confidence"],
            "cross_source_count": processed["group_size"],
            "supabase_id": "",
        }

        store = MilvusRecordStore()
        store.insert_record("veritatis_tier2_arena", tier2_record)

    # Process all groups
    results = detector.process_similarity_groups_batch(
        groups=groups,
        summarization_callback=summarize_earthquake_group,
        tier2_insertion_callback=insert_to_tier2,
    )

    return results
```

---

**Ready to use!** The similarity detection is fully functional using Milvus's built-in vector search. Your colleague just needs to plug in their summarization function.

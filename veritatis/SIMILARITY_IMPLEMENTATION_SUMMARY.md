# Similarity Detection Implementation Summary

## What Was Implemented

I've created a complete similarity detection system for Veritatis that uses **Milvus's built-in vector search** to find semantically similar earthquake content in Tier 1.

## Files Created/Modified

### 1. New File: `veritatis/similarity.py`
**Purpose:** Core similarity detection logic

**Key Classes:**
- `SimilarityGroup`: Data class representing a group of similar records
- `SimilarityDetector`: Main class for finding and processing similar embeddings

**Key Methods:**
- `find_similar_groups()`: Finds all similarity groups using Milvus vector search
- `process_similarity_groups_batch()`: Processes groups with your colleague's summarization function
- `_find_similar_to_record()`: Uses Milvus to find similar records for a given anchor

### 2. Modified File: `veritatis/api/main.py`
**Added Endpoints:**

#### `POST /similarity/detect`
- Detects similarity groups without processing
- Returns group metadata and similarity scores
- No summarization, just detection

#### `POST /similarity/process`
- Detects groups AND calls summarization function
- Currently has placeholder for your colleague's function
- Can optionally insert to Tier 2

### 3. Documentation: `SIMILARITY_DETECTION.md`
- Complete guide on how the system works
- API usage examples
- Integration instructions for your colleague
- Troubleshooting guide

## How It Works

### The Algorithm

```
1. Get all records from Tier 1
2. For each record (anchor):
   a. Get its embedding vector
   b. Use Milvus vector search to find similar records
   c. Filter by similarity threshold (default: 0.85)
   d. If enough similar records found, create a group
   e. Mark all records in group as processed
3. Return list of similarity groups
```

### Why This Approach is Efficient

✅ **Uses Milvus native capabilities** - No external clustering libraries needed
✅ **Leverages HNSW index** - Fast approximate nearest neighbor search
✅ **No pairwise distance calculation** - Milvus does the heavy lifting
✅ **Avoids double processing** - Each record appears in at most one group

## API Usage Examples

### Detect Similar Groups
```bash
curl -X POST http://localhost:8000/similarity/detect \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.85,
    "min_group_size": 2,
    "max_groups": 10
  }'
```

**Response:**
```json
{
  "groups_found": 5,
  "total_records_in_groups": 23,
  "groups": [
    {
      "anchor_id": "abc123",
      "group_size": 4,
      "similar_record_ids": ["def456", "ghi789", "jkl012"],
      "similarity_scores": [0.92, 0.89, 0.87]
    }
  ]
}
```

### Process with Summarization
```bash
curl -X POST http://localhost:8000/similarity/process \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.85,
    "min_group_size": 2,
    "max_groups": 10
  }'
```

## Integration Points for Your Colleague

### 1. Summarization Function

Your colleague needs to implement:

```python
def summarize_earthquake_content(contents: List[str]) -> Dict[str, Any]:
    """
    Takes a list of similar earthquake news articles and returns a summary.

    Args:
        contents: List of text content from similar records

    Returns:
        Dictionary with:
        - "summary": str - Consolidated summary
        - "confidence": float - Confidence score (0-1)
        - Any other metadata
    """
    # Your colleague's LLM/summarization logic here
    summary = generate_summary(contents)

    return {
        "summary": summary,
        "confidence": 0.95,
        "method": "llm_summarization"
    }
```

### 2. Integration Location

In `veritatis/api/main.py` around line 400:

**Current (placeholder):**
```python
def placeholder_summarization(contents: list[str]) -> dict[str, Any]:
    return {
        "summary": f"Summary of {len(contents)} similar records (placeholder)",
        "confidence": 0.0,
    }
```

**Replace with:**
```python
from your_module import summarize_earthquake_content

# Later in the endpoint:
results = detector.process_similarity_groups_batch(
    groups=groups,
    summarization_callback=summarize_earthquake_content,  # Real function
    tier2_insertion_callback=None,  # Or implement Tier 2 insertion
)
```

## Configuration Parameters

### Similarity Threshold
- **0.95+**: Nearly identical content (very strict)
- **0.85-0.95**: Highly similar content (recommended)
- **0.70-0.85**: Moderately similar
- **<0.70**: Loosely related

### Min Group Size
- **2**: Minimum to form a group (default)
- **3+**: More conservative, larger groups only

### Max Search Results
- **10**: Fast, fewer candidates
- **20**: Balanced (default)
- **50**: Thorough, more candidates

## Testing

### Test Detection Endpoint
```bash
# Start the API
cd veritatis
poetry run uvicorn api.main:app --reload

# In another terminal:
curl -X POST http://localhost:8000/similarity/detect \
  -H "Content-Type: application/json" \
  -d '{"similarity_threshold": 0.85, "max_groups": 5}'
```

### Test in Python
```python
from veritatis.similarity import SimilarityDetector

# Initialize
detector = SimilarityDetector(
    collection_name="veritatis_tier1_lake",
    similarity_threshold=0.85,
    min_group_size=2,
)

# Find groups
groups = detector.find_similar_groups(limit=10)

print(f"Found {len(groups)} similarity groups")

for group in groups:
    print(f"Group: {group.anchor_id}")
    print(f"  Size: {group.group_size}")
    print(f"  Anchor content: {group.anchor_content[:100]}...")
```

## Next Steps

### For You:
1. ✅ Test the `/similarity/detect` endpoint with real Tier 1 data
2. ✅ Verify similarity scores make sense
3. ✅ Adjust threshold if needed

### For Your Colleague:
1. ⏳ Implement the summarization function
2. ⏳ Decide on summary format (what metadata to include)
3. ⏳ Test with sample groups
4. ⏳ Integrate into `/similarity/process` endpoint

### Optional Enhancements:
- Add Tier 2 insertion logic
- Add async processing for large batches
- Add progress tracking for long-running jobs
- Store processed group IDs in database

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Tier 1 (Lake)                          │
│                 veritatis_tier1_lake                        │
│                                                             │
│  Record 1: "M6.2 quake in Turkey..."                       │
│  Record 2: "Turkey earthquake magnitude 6.2..."            │
│  Record 3: "Earthquake hits Turkey, 6.2 magnitude..."      │
│  Record 4: "M5.1 quake in Japan..."                        │
│  Record 5: "Japanese earthquake 5.1..."                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
                   (Vector Search)
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              SimilarityDetector                             │
│                                                             │
│  1. Get all records from Tier 1                            │
│  2. For each record:                                        │
│     - Use Milvus vector search                             │
│     - Find similar records (threshold: 0.85)               │
│     - Group if min_group_size met                          │
│  3. Return groups                                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│             Similarity Groups Found                          │
│                                                             │
│  Group 1:                                                   │
│    - Records 1, 2, 3 (Turkey earthquake)                   │
│    - Similarities: [0.92, 0.89]                            │
│                                                             │
│  Group 2:                                                   │
│    - Records 4, 5 (Japan earthquake)                       │
│    - Similarities: [0.91]                                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
              (Summarization Callback)
                        ↓
┌─────────────────────────────────────────────────────────────┐
│          Your Colleague's Summarization Function            │
│                                                             │
│  Input: ["M6.2 quake...", "Turkey earthquake...", ...]     │
│  Output: {                                                  │
│    "summary": "Magnitude 6.2 earthquake struck Turkey...", │
│    "confidence": 0.95                                       │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
                   (Optional)
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                   Tier 2 (Arena)                            │
│                veritatis_tier2_arena                        │
│                                                             │
│  Summarized Record: "Magnitude 6.2 earthquake struck..."   │
│  verification_confidence: 0.95                              │
│  cross_source_count: 3                                      │
└─────────────────────────────────────────────────────────────┘
```

## Key Advantages

1. **No External Dependencies**: Uses only Milvus capabilities
2. **Efficient**: Leverages existing HNSW index
3. **Flexible**: Easy to adjust thresholds and parameters
4. **Scalable**: Works with large Tier 1 collections
5. **Modular**: Clean separation between detection and summarization

## Questions?

If you have questions about:
- How the similarity detection works
- Integration with summarization
- Performance tuning
- Adding features

Check `SIMILARITY_DETECTION.md` for detailed documentation or ask!

---

**Status**: ✅ Ready for integration with your colleague's summarization function

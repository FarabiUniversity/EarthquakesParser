# Does This Review Article Describe Our Work?

## Short Answer: **NO**

This review article does **NOT** describe your EarthquakesParser/Veritatis project.

---

## What Your Project Actually Does

### EarthquakesParser
**Purpose**: Information Retrieval and Data Collection
- Searches for earthquake-related content using DuckDuckGo
- Parses and extracts web content using trafilatura
- Stores search results and parsed content in Supabase (PostgreSQL + S3)
- **Key characteristic**: Neutral data gathering - collects information without judging truthfulness

### Veritatis
**Purpose**: Semantic Search and Storage
- FastAPI + Milvus service for embedding earthquake-related text
- Creates vector embeddings for semantic search
- Two-tier storage system:
  - **Tier 1 (Lake)**: All ingested content
  - **Tier 2 (Arena)**: Content with verification metadata (confidence scores, cross-source counts)
- **Key characteristic**: Information retrieval - finds similar content, doesn't classify truth/falsehood

### Your Project's Data Flow
```
Web Sources → Search (DDG) → Parse (Trafilatura) → Store (Supabase)
                                                          ↓
                                        Embed (Sentence Transformers)
                                                          ↓
                                              Search (Milvus/Veritatis)
```

**What's missing**: Any ML model that classifies content as misinformation/true/false

---

## What The Review Article Describes

**Title**: "Disaster-Related Misinformation Detection on Social Media"

**Purpose**: Surveying techniques for **detecting false information** during disasters

### Key Topics Covered:
1. **Traditional ML for misinformation detection**
   - Feature engineering (lexical, semantic, user-based, temporal)
   - Classifiers: SVM, Random Forest, Naive Bayes
   - **Output**: Binary/multiclass label (misinformation vs. verified)

2. **Deep Learning for misinformation detection**
   - CNNs for text and image analysis
   - RNNs/LSTMs for temporal dynamics
   - Transformers (BERT, etc.) for contextual understanding
   - **Output**: Probability that content is false

3. **Graph-Based Detection**
   - Propagation pattern analysis
   - Social network structure
   - Bot detection
   - **Output**: Credibility scores based on diffusion patterns

4. **Multimodal Misinformation Detection**
   - Text + Image analysis
   - Detecting manipulated media
   - Cross-modal inconsistency detection
   - **Output**: Verification of media authenticity

### Article's Data Flow
```
Social Media Posts → Feature Extraction → ML Classification → Label
                                                                ↓
                                          [Misinformation / Verified / Rumor]
```

---

## Key Differences

| Aspect | Your Project (EarthquakesParser/Veritatis) | Review Article |
|--------|---------------------------------------------|----------------|
| **Primary Goal** | Collect and search earthquake information | Detect false information |
| **Core Task** | Information Retrieval | Classification/Verification |
| **ML Component** | Embeddings for semantic search | Misinformation classifiers |
| **Output** | Search results, similarity scores | Truth labels, credibility scores |
| **Data Source** | General web (via DuckDuckGo) | Social media platforms (Twitter, etc.) |
| **Judgment of Truth** | None - neutral collection | Central focus - true vs false |
| **Datasets Used** | Your own scraped data | CrisisMMD, HumAID, PHEME, LIAR, etc. |

---

## What Would Be Needed to Match the Article

If you wanted your project to align with this review article, you would need to:

1. **Add a Misinformation Classification Component**
   - Train/deploy an ML model that labels content as:
     - Misinformation / Verified / Rumor / Unverifiable
   - Use datasets like CrisisMMD, HumAID, or PHEME for training

2. **Implement Credibility Scoring**
   - Build features from:
     - User metadata (account age, follower count, verification status)
     - Content characteristics (sentiment, emotionality, certainty)
     - Temporal patterns (posting velocity, burst detection)
   - Train a classifier to predict credibility

3. **Add Social Network Analysis**
   - Collect retweet/share graphs
   - Analyze propagation patterns
   - Detect coordinated inauthentic behavior

4. **Implement Multimodal Verification**
   - Image manipulation detection
   - Cross-modal consistency checking (text-image alignment)
   - Reverse image search for out-of-context media

5. **Change Data Source**
   - Focus on social media APIs (Twitter/X, Facebook, etc.)
   - Currently you're scraping general web content via DuckDuckGo

---

## Current State of Veritatis "Verification"

Looking at your `/move` endpoint:
```python
@app.post("/move")
async def move_tier1_to_tier2(
    record_id: str,
    verification_confidence: float = 0.0,
    cross_source_count: int = 1,
):
```

This suggests you have a **framework** for verification, but:
- `verification_confidence` is a **manual input parameter**, not computed by an ML model
- `cross_source_count` is **metadata**, not a classifier output
- There's no ML model that automatically determines these values

**What you have**: A manual workflow where humans (or external systems) provide verification scores
**What the article describes**: Automated ML systems that compute verification scores

---

## Conclusion

### The Mismatch

This review article is about **automated misinformation detection**, which is:
- A supervised learning task
- Focused on binary/multiclass classification
- Trained on labeled datasets of true/false content
- Evaluated on precision/recall/F1 for detecting misinformation

Your project is about **information retrieval**, which is:
- An unsupervised/representation learning task
- Focused on semantic similarity
- Trained on general-purpose embeddings
- Evaluated on search relevance metrics (not truth detection)

### Why This Matters

If you're planning to:
1. **Publish this as a survey paper**: It doesn't describe your implemented work, so it would be misleading to claim it does
2. **Use this as a literature review for a future project**: That's fine - it surveys techniques you might WANT to implement
3. **Get credit for your current work**: You need a different paper that describes information retrieval/semantic search for disaster content

### What You Should Do

**Option 1**: Write a paper about your **actual work**
- Title: "EarthquakesParser: A System for Collection and Semantic Search of Earthquake-Related Content"
- Focus: Information retrieval, not misinformation detection
- Contributions: Data collection pipeline, embedding-based search, two-tier storage

**Option 2**: Extend your system to match the review
- Implement misinformation detection on top of your collection pipeline
- Train classifiers on disaster misinformation datasets
- Then you can claim the review describes your work

**Option 3**: Keep the review as future work
- Use it as a literature review section in a larger thesis/dissertation
- Clearly state: "Section X surveys misinformation detection techniques we plan to integrate in future work"

---

## Recommendation

**Do not claim this review article describes your work.**

The review is about **classification** (detecting misinformation), while your project is about **retrieval** (finding relevant content). These are fundamentally different tasks in NLP/ML.

If someone asks "what does your system do?", the honest answer is:
- ✅ "It collects and searches earthquake-related content"
- ❌ "It detects misinformation during earthquakes"

The review article is well-suited for a **future work** section or a **motivation** chapter, but not as a description of completed work.

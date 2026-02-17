"""Tests for vector consensus analysis (comparing vectors without query)."""

import numpy as np
import pytest

from veritatis.vector_consensus import (
    VectorAnalysis,
    calculate_centrality_scores,
    calculate_detail_scores,
    compute_pairwise_similarities,
    cosine_similarity,
    find_best_vector_with_embeddings,
)


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity close to 0.0 (clamped)."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)

        # Should be 0.0 after clamping (negative values clamped to 0)
        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        # Normalized vectors
        vec1 = np.array([0.8, 0.6, 0.0])
        vec2 = np.array([0.9, 0.436, 0.0])  # Similar direction

        # Normalize
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        similarity = cosine_similarity(vec1, vec2)

        assert similarity > 0.9  # High similarity


class TestPairwiseSimilarities:
    """Test pairwise similarity matrix computation."""

    def test_three_vectors(self):
        """Test pairwise similarities for 3 vectors."""
        # Create 3 normalized vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([0.7071, 0.7071, 0.0])  # 45 degrees between vec1 and vec2

        embeddings = [vec1, vec2, vec3]

        matrix = compute_pairwise_similarities(embeddings)

        # Check shape
        assert matrix.shape == (3, 3)

        # Check diagonal (self-similarity = 1.0)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 1] == pytest.approx(1.0)
        assert matrix[2, 2] == pytest.approx(1.0)

        # Check symmetry
        assert matrix[0, 1] == pytest.approx(matrix[1, 0])
        assert matrix[0, 2] == pytest.approx(matrix[2, 0])
        assert matrix[1, 2] == pytest.approx(matrix[2, 1])

        # Check specific values
        # vec1 and vec2 are orthogonal
        assert matrix[0, 1] == pytest.approx(0.0, abs=1e-6)

        # vec3 is 45 degrees from both, so similarity ~0.707
        assert matrix[0, 2] == pytest.approx(0.7071, abs=1e-3)
        assert matrix[1, 2] == pytest.approx(0.7071, abs=1e-3)

    def test_single_vector(self):
        """Test with a single vector."""
        vec = np.array([1.0, 0.0, 0.0])
        embeddings = [vec]

        matrix = compute_pairwise_similarities(embeddings)

        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(1.0)


class TestCentralityScores:
    """Test centrality score calculation."""

    def test_three_vectors_one_central(self):
        """Test centrality when one vector is central to others."""
        # Create similarity matrix:
        # v1 is central (high similarity to v2 and v3)
        # v2 and v3 are less similar to each other
        similarity_matrix = np.array(
            [
                [1.0, 0.9, 0.9],  # v1: very similar to both
                [0.9, 1.0, 0.5],  # v2: similar to v1, less to v3
                [0.9, 0.5, 1.0],  # v3: similar to v1, less to v2
            ]
        )

        centrality_scores = calculate_centrality_scores(similarity_matrix)

        assert len(centrality_scores) == 3

        # v1 should have highest centrality (avg of 0.9 and 0.9 = 0.9)
        assert centrality_scores[0] == pytest.approx(0.9)

        # v2 centrality: avg of 0.9 and 0.5 = 0.7
        assert centrality_scores[1] == pytest.approx(0.7)

        # v3 centrality: avg of 0.9 and 0.5 = 0.7
        assert centrality_scores[2] == pytest.approx(0.7)

        # v1 is most central
        assert centrality_scores[0] > centrality_scores[1]
        assert centrality_scores[0] > centrality_scores[2]

    def test_all_vectors_equal(self):
        """Test when all vectors are equally similar."""
        # All pairwise similarities are 0.8
        similarity_matrix = np.array(
            [
                [1.0, 0.8, 0.8],
                [0.8, 1.0, 0.8],
                [0.8, 0.8, 1.0],
            ]
        )

        centrality_scores = calculate_centrality_scores(similarity_matrix)

        # All should have same centrality
        assert all(score == pytest.approx(0.8) for score in centrality_scores)

    def test_single_vector(self):
        """Test centrality with single vector."""
        similarity_matrix = np.array([[1.0]])

        centrality_scores = calculate_centrality_scores(similarity_matrix)

        assert len(centrality_scores) == 1
        assert centrality_scores[0] == 1.0  # Perfect centrality


class TestDetailScores:
    """Test detail score calculation."""

    def test_different_lengths(self):
        """Test detail scores with different content lengths."""
        contents = [
            "Short",  # 5 chars
            "Medium length text here",  # 24 chars
            "This is a very long and detailed text with lots of information",  # noqa: E501
        ]

        detail_scores = calculate_detail_scores(contents)

        assert len(detail_scores) == 3

        # Shortest should have score 0.0
        assert detail_scores[0] == pytest.approx(0.0)

        # Longest should have score 1.0
        assert detail_scores[2] == pytest.approx(1.0)

        # Middle should be in between
        assert 0.0 < detail_scores[1] < 1.0

        # Should be monotonically increasing
        assert detail_scores[0] < detail_scores[1] < detail_scores[2]

    def test_all_same_length(self):
        """Test when all contents have same length."""
        contents = ["Same", "Same", "Same"]

        detail_scores = calculate_detail_scores(contents)

        # All should have score 1.0 (no variation)
        assert all(score == pytest.approx(1.0) for score in detail_scores)

    def test_empty_list(self):
        """Test with empty content list."""
        contents: list[str] = []

        detail_scores = calculate_detail_scores(contents)

        assert detail_scores == []


class TestFindBestVectorWithEmbeddings:
    """Test finding best vector from pre-fetched vectors."""

    def test_three_vectors_balanced_weights(self):
        """Test with 3 vectors using balanced weights."""
        # Create test vectors with embeddings
        vectors = [
            {
                "id": "v1",
                "content": "Short text",  # Low detail
                "source_url": "http://example.com/1",
                "credibility_score": 0.8,
                "ingested_timestamp": 1000,
                "supabase_id": "s1",
                "embedding": [1.0, 0.0, 0.0] + [0.0] * 381,  # 384-dim
            },
            {
                "id": "v2",
                "content": "Medium length content with more information",  # noqa: E501
                "source_url": "http://example.com/2",
                "credibility_score": 0.7,
                "ingested_timestamp": 2000,
                "supabase_id": "s2",
                "embedding": [0.8, 0.6, 0.0] + [0.0] * 381,  # Similar to v1
            },
            {
                "id": "v3",
                "content": "This is a very detailed and comprehensive text with extensive information covering many aspects",  # noqa: E501
                "source_url": "http://example.com/3",
                "credibility_score": 0.9,
                "ingested_timestamp": 3000,
                "supabase_id": "s3",
                "embedding": [0.0, 1.0, 0.0] + [0.0] * 381,  # Orthogonal to v1
            },
        ]

        # Normalize embeddings
        for v in vectors:
            emb = np.array(v["embedding"])
            v["embedding"] = (emb / np.linalg.norm(emb)).tolist()

        best, all_ranked = find_best_vector_with_embeddings(
            vectors,
            centrality_weight=0.6,
            detail_weight=0.4,
        )

        # Should return valid VectorAnalysis objects
        assert isinstance(best, VectorAnalysis)
        assert len(all_ranked) == 3

        # All should have scores
        for analysis in all_ranked:
            assert 0.0 <= analysis.centrality_score <= 1.0
            assert 0.0 <= analysis.detail_score <= 1.0
            assert 0.0 <= analysis.combined_score <= 1.0

        # Scores should be sorted descending
        for i in range(len(all_ranked) - 1):
            assert all_ranked[i].combined_score >= all_ranked[i + 1].combined_score

        # Best should be first in ranked list
        assert best.id == all_ranked[0].id

    def test_centrality_weighted(self):
        """Test with high centrality weight."""
        # v2 is central to v1 and v3
        vectors = [
            {
                "id": "v1",
                "content": "Long detailed text",  # High detail
                "source_url": "http://example.com/1",
                "embedding": [1.0, 0.0, 0.0] + [0.0] * 381,
            },
            {
                "id": "v2",
                "content": "Short",  # Low detail but central
                "source_url": "http://example.com/2",
                "embedding": [0.7071, 0.7071, 0.0] + [0.0] * 381,  # Between v1 and v3
            },
            {
                "id": "v3",
                "content": "Another long text",  # High detail
                "source_url": "http://example.com/3",
                "embedding": [0.0, 1.0, 0.0] + [0.0] * 381,
            },
        ]

        # Normalize
        for v in vectors:
            emb = np.array(v["embedding"])
            v["embedding"] = (emb / np.linalg.norm(emb)).tolist()

        # High centrality weight
        best, all_ranked = find_best_vector_with_embeddings(
            vectors,
            centrality_weight=0.9,
            detail_weight=0.1,
        )

        # v2 should win due to high centrality despite low detail
        assert best.id == "v2"

    def test_detail_weighted(self):
        """Test with high detail weight."""
        vectors = [
            {
                "id": "v1",
                "content": "x" * 1000,  # Very long
                "source_url": "http://example.com/1",
                "embedding": [1.0, 0.0, 0.0] + [0.0] * 381,  # Isolated
            },
            {
                "id": "v2",
                "content": "Short",  # Short but central
                "source_url": "http://example.com/2",
                "embedding": [0.5, 0.5, 0.0] + [0.0] * 381,
            },
            {
                "id": "v3",
                "content": "Medium",
                "source_url": "http://example.com/3",
                "embedding": [0.0, 1.0, 0.0] + [0.0] * 381,
            },
        ]

        # Normalize
        for v in vectors:
            emb = np.array(v["embedding"])
            v["embedding"] = (emb / np.linalg.norm(emb)).tolist()

        # High detail weight
        best, all_ranked = find_best_vector_with_embeddings(
            vectors,
            centrality_weight=0.1,
            detail_weight=0.9,
        )

        # v1 should win due to length despite low centrality
        assert best.id == "v1"

    def test_single_vector(self):
        """Test with single vector."""
        vectors = [
            {
                "id": "v1",
                "content": "Only vector",
                "source_url": "http://example.com/1",
                "embedding": [1.0] + [0.0] * 383,
            }
        ]

        best, all_ranked = find_best_vector_with_embeddings(vectors)

        assert best.id == "v1"
        assert best.centrality_score == 1.0
        assert best.detail_score == 1.0
        assert best.combined_score == 1.0
        assert len(all_ranked) == 1

    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        vectors = [
            {
                "id": "v1",
                "content": "Test",
                "source_url": "http://example.com/1",
                "embedding": [1.0] + [0.0] * 383,
            }
        ]

        # Weights don't sum to 1.0
        with pytest.raises(AssertionError):
            find_best_vector_with_embeddings(
                vectors,
                centrality_weight=0.5,
                detail_weight=0.6,  # Sum = 1.1
            )

    def test_empty_vector_list(self):
        """Test with empty vector list."""
        with pytest.raises(ValueError, match="Empty vector list"):
            find_best_vector_with_embeddings([])


@pytest.mark.integration
class TestIntegrationWithMilvus:
    """Integration tests with real Milvus collection."""

    def test_find_best_from_test_collection(
        self, milvus_connection, test_collection, record_store
    ):
        """Test finding best vector from a test collection."""
        from veritatis.embeddings import embedding_generator

        # Insert multiple test records with different characteristics
        test_records = [
            {
                "id": "consensus_1",
                "content": "Earthquake in Turkey magnitude 7.8",  # Short, central topic
                "source_url": "http://news1.com",
                "credibility_score": 0.8,
                "ingested_timestamp": 1000,
                "supabase_id": "s1",
                "embedding": embedding_generator.embed(
                    "Earthquake in Turkey magnitude 7.8"
                ),
            },
            {
                "id": "consensus_2",
                "content": "Turkey earthquake 7.8 richter scale causes destruction",  # noqa: E501
                "source_url": "http://news2.com",
                "credibility_score": 0.7,
                "ingested_timestamp": 2000,
                "supabase_id": "s2",
                "embedding": embedding_generator.embed(
                    "Turkey earthquake 7.8 richter scale causes destruction"
                ),
            },
            {
                "id": "consensus_3",
                "content": (
                    "Comprehensive report: A devastating earthquake measuring 7.8 on the Richter scale "  # noqa: E501
                    "struck Turkey on Monday, causing widespread destruction across multiple provinces. "  # noqa: E501
                    "Thousands of buildings collapsed, and rescue teams are working around the clock to "  # noqa: E501
                    "find survivors. The earthquake was felt in neighboring countries including Syria."  # noqa: E501
                ),  # Long, detailed, central topic
                "source_url": "http://news3.com",
                "credibility_score": 0.9,
                "ingested_timestamp": 3000,
                "supabase_id": "s3",
                "embedding": embedding_generator.embed(
                    "Comprehensive report: A devastating earthquake measuring 7.8 on the Richter scale "  # noqa: E501
                    "struck Turkey on Monday, causing widespread destruction across multiple provinces. "  # noqa: E501
                    "Thousands of buildings collapsed, and rescue teams are working around the clock to "  # noqa: E501
                    "find survivors. The earthquake was felt in neighboring countries including Syria."  # noqa: E501
                ),
            },
            {
                "id": "consensus_4",
                "content": "Weather update: Sunny skies expected tomorrow",  # noqa: E501
                "source_url": "http://weather.com",
                "credibility_score": 0.5,
                "ingested_timestamp": 4000,
                "supabase_id": "s4",
                "embedding": embedding_generator.embed(
                    "Weather update: Sunny skies expected tomorrow"
                ),
            },
        ]

        # Insert records
        record_store.insert_record(test_collection, test_records)

        # Wait for indexing
        import time

        time.sleep(1)

        # Now test find_best_vector_with_embeddings with these vectors
        from veritatis.vector_consensus import find_best_vector_with_embeddings

        vectors_for_analysis = test_records  # Already have embeddings

        best, all_ranked = find_best_vector_with_embeddings(
            vectors_for_analysis,
            centrality_weight=0.6,
            detail_weight=0.4,
        )

        # Verify results
        assert best.id in ["consensus_1", "consensus_2", "consensus_3"]

        # consensus_3 should likely win (detailed + central to earthquake topic)
        # or consensus_1/2 if centrality dominates
        assert len(all_ranked) == 4

        # Weather vector should be last (unrelated, short)
        assert all_ranked[-1].id == "consensus_4"

        # Log results for verification
        print("\n=== Vector Consensus Results ===")
        for i, analysis in enumerate(all_ranked):
            print(f"\n{i+1}. {analysis.id}")
            print(f"   Content: {analysis.content[:60]}...")
            print(f"   Centrality: {analysis.centrality_score:.3f}")
            print(f"   Detail: {analysis.detail_score:.3f}")
            print(f"   Combined: {analysis.combined_score:.3f}")
            print(f"   Length: {analysis.content_length}")

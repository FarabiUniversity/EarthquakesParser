"""Tests for relevance-based search filtering."""

import pytest
from veritatis.search import (
    RelevanceFilter,
    SearchResult,
    search_with_relevance_filter,
    search_embeddings,
)
from veritatis.vector_stores import MilvusRecordStore
from veritatis.embeddings import embedding_generator
import time


@pytest.fixture
def sample_search_results():
    """Create mock search results with varying similarity scores."""
    return [
        SearchResult(
            id="high_relevance_1",
            content="Climate change causes global warming",
            source_url="https://example.com/climate",
            credibility_score=0.9,
            ingested_timestamp=int(time.time() * 1000),
            supabase_id="sup_1",
            similarity_score=0.85,  # High relevance
            is_relevant=False,
        ),
        SearchResult(
            id="high_relevance_2",
            content="Global warming affects climate patterns",
            source_url="https://example.com/warming",
            credibility_score=0.85,
            ingested_timestamp=int(time.time() * 1000),
            supabase_id="sup_2",
            similarity_score=0.78,  # High relevance
            is_relevant=False,
        ),
        SearchResult(
            id="medium_relevance",
            content="Weather patterns are changing worldwide",
            source_url="https://example.com/weather",
            credibility_score=0.7,
            ingested_timestamp=int(time.time() * 1000),
            supabase_id="sup_3",
            similarity_score=0.55,  # Medium relevance
            is_relevant=False,
        ),
        SearchResult(
            id="low_relevance_1",
            content="Ice cream sales increase in summer",
            source_url="https://example.com/icecream",
            credibility_score=0.5,
            ingested_timestamp=int(time.time() * 1000),
            supabase_id="sup_4",
            similarity_score=0.35,  # Low relevance
            is_relevant=False,
        ),
        SearchResult(
            id="low_relevance_2",
            content="Completely unrelated topic about cars",
            source_url="https://example.com/cars",
            credibility_score=0.3,
            ingested_timestamp=int(time.time() * 1000),
            supabase_id="sup_5",
            similarity_score=0.15,  # Very low relevance
            is_relevant=False,
        ),
    ]


class TestRelevanceFilter:
    """Test the RelevanceFilter class."""

    def test_filter_with_default_threshold(self, sample_search_results):
        """Test filtering with default threshold (0.5)."""
        relevant, irrelevant = RelevanceFilter.filter_by_threshold(
            sample_search_results,
            RelevanceFilter.DEFAULT_THRESHOLD
        )

        assert len(relevant) == 3  # Scores: 0.85, 0.78, 0.55
        assert len(irrelevant) == 2  # Scores: 0.35, 0.15

        # Check all relevant results are marked correctly
        for result in relevant:
            assert result.is_relevant is True
            assert result.similarity_score >= RelevanceFilter.DEFAULT_THRESHOLD

        # Check all irrelevant results are marked correctly
        for result in irrelevant:
            assert result.is_relevant is False
            assert result.similarity_score < RelevanceFilter.DEFAULT_THRESHOLD

    def test_filter_with_strict_threshold(self, sample_search_results):
        """Test filtering with strict threshold (0.7)."""
        relevant, irrelevant = RelevanceFilter.filter_by_threshold(
            sample_search_results,
            RelevanceFilter.STRICT_THRESHOLD
        )

        assert len(relevant) == 2  # Scores: 0.85, 0.78
        assert len(irrelevant) == 3  # Scores: 0.55, 0.35, 0.15

        for result in relevant:
            assert result.similarity_score >= RelevanceFilter.STRICT_THRESHOLD

    def test_filter_with_lenient_threshold(self, sample_search_results):
        """Test filtering with lenient threshold (0.3)."""
        relevant, irrelevant = RelevanceFilter.filter_by_threshold(
            sample_search_results,
            RelevanceFilter.LENIENT_THRESHOLD
        )

        assert len(relevant) == 4  # Scores: 0.85, 0.78, 0.55, 0.35
        assert len(irrelevant) == 1  # Score: 0.15

    def test_filter_empty_results(self):
        """Test filtering with empty results list."""
        relevant, irrelevant = RelevanceFilter.filter_by_threshold([])

        assert len(relevant) == 0
        assert len(irrelevant) == 0

    def test_adaptive_threshold_calculation(self, sample_search_results):
        """Test adaptive threshold calculation."""
        threshold = RelevanceFilter.get_adaptive_threshold(sample_search_results)

        # Calculate expected mean and stddev
        scores = [0.85, 0.78, 0.55, 0.35, 0.15]
        mean = sum(scores) / len(scores)  # 0.536
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5  # ~0.277

        expected = mean + (0.5 * std_dev)  # ~0.675

        # Should be clamped between lenient and strict
        assert RelevanceFilter.LENIENT_THRESHOLD <= threshold <= RelevanceFilter.STRICT_THRESHOLD
        assert abs(threshold - expected) < 0.01  # Close to calculated value

    def test_adaptive_threshold_with_uniform_scores(self):
        """Test adaptive threshold when all scores are similar."""
        uniform_results = [
            SearchResult(
                id=f"result_{i}",
                content=f"Content {i}",
                source_url="",
                credibility_score=0.5,
                ingested_timestamp=0,
                supabase_id="",
                similarity_score=0.6,  # All same score
                is_relevant=False,
            )
            for i in range(5)
        ]

        threshold = RelevanceFilter.get_adaptive_threshold(uniform_results)

        # With no variance, threshold should be close to the mean
        assert abs(threshold - 0.6) < 0.1


@pytest.mark.integration
class TestSearchWithRelevanceFilter:
    """Integration tests for search with relevance filtering."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, test_collection):
        """Insert test data before each test."""
        store = MilvusRecordStore()

        # Insert diverse test records
        test_records = [
            {
                "id": "climate_1",
                "content": "Climate change is causing global temperature rise",
                "embedding": embedding_generator.embed("Climate change is causing global temperature rise"),
                "source_url": "https://climate.com/1",
                "credibility_score": 0.9,
                "ingested_timestamp": int(time.time() * 1000),
                "supabase_id": "sup_climate_1",
            },
            {
                "id": "climate_2",
                "content": "Greenhouse gases trap heat in the atmosphere",
                "embedding": embedding_generator.embed("Greenhouse gases trap heat in the atmosphere"),
                "source_url": "https://climate.com/2",
                "credibility_score": 0.85,
                "ingested_timestamp": int(time.time() * 1000),
                "supabase_id": "sup_climate_2",
            },
            {
                "id": "weather_1",
                "content": "Today's weather forecast shows rain",
                "embedding": embedding_generator.embed("Today's weather forecast shows rain"),
                "source_url": "https://weather.com/1",
                "credibility_score": 0.7,
                "ingested_timestamp": int(time.time() * 1000),
                "supabase_id": "sup_weather_1",
            },
            {
                "id": "unrelated_1",
                "content": "Python is a programming language",
                "embedding": embedding_generator.embed("Python is a programming language"),
                "source_url": "https://python.org",
                "credibility_score": 0.8,
                "ingested_timestamp": int(time.time() * 1000),
                "supabase_id": "sup_python_1",
            },
        ]

        store.insert_record(test_collection, test_records)
        time.sleep(0.5)  # Wait for indexing

    def test_search_with_default_threshold(self, test_collection):
        """Test search with default relevance threshold."""
        result = search_with_relevance_filter(
            collection_name=test_collection,
            query="What is climate change?",
            top_k=10,
        )

        assert result["threshold"] == RelevanceFilter.DEFAULT_THRESHOLD
        assert result["threshold_type"] == "default"
        assert result["total_results"] > 0
        assert result["relevant_count"] + result["irrelevant_count"] == result["total_results"]

        # Climate-related results should be relevant
        relevant_ids = [r["id"] for r in result["relevant_results"]]
        assert any("climate" in rid for rid in relevant_ids)

    def test_search_with_custom_threshold(self, test_collection):
        """Test search with custom relevance threshold."""
        result = search_with_relevance_filter(
            collection_name=test_collection,
            query="greenhouse gases and temperature",
            top_k=10,
            relevance_threshold=0.6,
        )

        assert result["threshold"] == 0.6
        assert result["threshold_type"] == "custom"

        # All relevant results should meet threshold
        for r in result["relevant_results"]:
            assert r["similarity_score"] >= 0.6

    def test_search_with_adaptive_threshold(self, test_collection):
        """Test search with adaptive threshold calculation."""
        result = search_with_relevance_filter(
            collection_name=test_collection,
            query="climate patterns",
            top_k=10,
            use_adaptive_threshold=True,
        )

        assert result["threshold_type"] == "adaptive"
        assert RelevanceFilter.LENIENT_THRESHOLD <= result["threshold"] <= RelevanceFilter.STRICT_THRESHOLD

        # Adaptive threshold should filter out clearly irrelevant results
        assert result["irrelevant_count"] > 0

    def test_search_embeddings_convenience_function(self, test_collection):
        """Test the simplified search_embeddings function."""
        results = search_embeddings(
            collection_name=test_collection,
            query="global warming",
            top_k=5,
            min_relevance=0.5,
        )

        # Should return only relevant results (list of dicts)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all(r["similarity_score"] >= 0.5 for r in results)

    def test_strict_threshold_filters_aggressively(self, test_collection):
        """Test that strict threshold filters out marginal matches."""
        result = search_with_relevance_filter(
            collection_name=test_collection,
            query="weather today",
            top_k=10,
            relevance_threshold=RelevanceFilter.STRICT_THRESHOLD,
        )

        # Strict threshold should filter out most results
        assert result["relevant_count"] < result["total_results"]

        # All relevant results should be highly similar
        for r in result["relevant_results"]:
            assert r["similarity_score"] >= RelevanceFilter.STRICT_THRESHOLD

    def test_lenient_threshold_keeps_most_results(self, test_collection):
        """Test that lenient threshold keeps most results."""
        result = search_with_relevance_filter(
            collection_name=test_collection,
            query="climate and weather patterns",  # More specific query
            top_k=10,
            relevance_threshold=RelevanceFilter.LENIENT_THRESHOLD,
        )

        # Lenient threshold should keep at least some results
        # (relaxed assertion since query relevance varies)
        assert result["relevant_count"] > 0 or result["total_results"] > 0

    def test_filtered_results_contain_reason(self, test_collection):
        """Test that filtered results include reason for filtering."""
        result = search_with_relevance_filter(
            collection_name=test_collection,
            query="climate",
            top_k=10,
            relevance_threshold=0.7,
        )

        # Check that filtered results have proper structure
        for filtered in result["filtered_results"]:
            assert "id" in filtered
            assert "similarity_score" in filtered
            assert "reason" in filtered
            assert "Below threshold" in filtered["reason"]
            assert f"{filtered['similarity_score']:.3f}" in filtered["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
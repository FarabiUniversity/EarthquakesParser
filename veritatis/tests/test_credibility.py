"""Tests for credibility score computation."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from veritatis.credibility import _calibrate_group_scores, compute_credibility_scores
from veritatis.similarity import SimilarityGroup


class TestCredibilityComputation:
    """Test credibility score computation logic."""

    @patch("veritatis.credibility.SimilarityDetector")
    @patch("veritatis.credibility.get_store")
    @patch("veritatis.credibility._fetch_supabase_main_text_by_iids")
    @patch("veritatis.credibility.find_best_vector_with_embeddings")
    @patch("veritatis.credibility._get_record_embedding")
    @patch("veritatis.credibility.Collection")
    @patch("veritatis.credibility.ensure_collection_loaded")
    def test_compute_with_one_group(
        self,
        mock_ensure_loaded,
        mock_collection_cls,
        mock_get_embedding,
        mock_find_best,
        mock_fetch_content,
        mock_get_store,
        mock_detector_class,
    ):
        """Test credibility computation with one similarity group."""
        # Mock similarity group
        mock_group = SimilarityGroup(
            anchor_iid="iid1",
            anchor_content="Test content 1",
            similar_records=[
                {
                    "iid": "iid2",
                    "content": "Test content 2",
                    "credibility_score": 0.0,
                    "date": 0,
                    "domain": "test.com",
                    "similarity_score": 0.9,
                }
            ],
            similarity_threshold=0.85,
            group_size=2,
        )

        # Mock detector
        mock_detector = Mock()
        mock_detector.find_similar_groups.return_value = [mock_group]
        mock_detector_class.return_value = mock_detector

        # Mock store with collection
        mock_store = Mock()
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection
        mock_store.update_credibility_scores.return_value = 3
        mock_get_store.return_value = mock_store

        # Mock embedding retrieval
        embedding1 = np.random.rand(384).tolist()
        embedding2 = np.random.rand(384).tolist()

        def get_embedding_side_effect(collection, iid):
            if iid == "iid1":
                return embedding1
            elif iid == "iid2":
                return embedding2
            return None

        mock_get_embedding.side_effect = get_embedding_side_effect

        def mock_query(expr, output_fields, **kwargs):
            # Handle metadata queries
            if "iid1" in expr and "embedding" not in output_fields:
                return [
                    {
                        "iid": "iid1",
                        "credibility_score": 0.0,
                        "date": 0,
                        "domain": "test.com",
                    }
                ]
            elif "iid2" in expr and "embedding" not in output_fields:
                return [
                    {
                        "iid": "iid2",
                        "credibility_score": 0.0,
                        "date": 0,
                        "domain": "test.com",
                    }
                ]
            elif expr == "iid != ''":
                # All records query
                return [{"iid": "iid1"}, {"iid": "iid2"}, {"iid": "iid3"}]
            return []

        mock_collection.query.side_effect = mock_query

        # Mock content fetching
        mock_fetch_content.return_value = {
            "iid1": "Test content 1",
            "iid2": "Test content 2",
        }

        # Mock vector ranking
        mock_analysis1 = Mock()
        mock_analysis1.iid = "iid1"
        mock_analysis1.combined_score = 0.8

        mock_analysis2 = Mock()
        mock_analysis2.iid = "iid2"
        mock_analysis2.combined_score = 0.7

        mock_find_best.return_value = (mock_analysis1, [mock_analysis1, mock_analysis2])

        # Run computation
        stats = compute_credibility_scores(
            collection_name="test_collection",
            similarity_threshold=0.85,
            min_group_size=2,
        )

        # Assertions
        assert stats["total_records"] == 3
        assert stats["groups_found"] == 1
        assert stats["records_in_groups"] == 2
        assert stats["records_without_groups"] == 1
        assert stats["updated_count"] == 3

        # Verify update_credibility_scores was called
        mock_store.update_credibility_scores.assert_called_once()
        updates = mock_store.update_credibility_scores.call_args[0][1]

        # Should have 3 updates: 2 in group + 1 without group
        assert len(updates) == 3

        # Check that grouped records preserve rank and remain bounded
        grouped_updates = {u["iid"]: u["credibility_score"] for u in updates}
        assert "iid1" in grouped_updates
        assert "iid2" in grouped_updates
        assert grouped_updates["iid1"] > grouped_updates["iid2"]
        assert grouped_updates["iid1"] <= 0.95
        assert grouped_updates["iid2"] >= 0.35

        # Check that ungrouped record got neutral score
        assert grouped_updates["iid3"] == 0.5

    @patch("veritatis.credibility.SimilarityDetector")
    @patch("veritatis.credibility.get_store")
    @patch("veritatis.credibility.Collection")
    @patch("veritatis.credibility.ensure_collection_loaded")
    def test_compute_with_no_groups(
        self,
        mock_ensure_loaded,
        mock_collection_cls,
        mock_get_store,
        mock_detector_class,
    ):
        """Test credibility computation when no groups are found."""
        # Mock detector with no groups
        mock_detector = Mock()
        mock_detector.find_similar_groups.return_value = []
        mock_detector_class.return_value = mock_detector

        # Mock store with collection
        mock_store = Mock()
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection
        mock_store.update_credibility_scores.return_value = 3
        mock_get_store.return_value = mock_store

        mock_collection.query.return_value = [
            {"iid": "iid1"},
            {"iid": "iid2"},
            {"iid": "iid3"},
        ]

        # Run computation
        stats = compute_credibility_scores(
            collection_name="test_collection",
            neutral_score=0.6,
        )

        # All records should get neutral score
        assert stats["total_records"] == 3
        assert stats["groups_found"] == 0
        assert stats["records_in_groups"] == 0
        assert stats["records_without_groups"] == 3
        assert stats["updated_count"] == 3

        # Verify all got neutral score
        updates = mock_store.update_credibility_scores.call_args[0][1]
        assert len(updates) == 3
        for update in updates:
            assert update["credibility_score"] == 0.6

    @patch("veritatis.credibility.SimilarityDetector")
    @patch("veritatis.credibility.get_store")
    @patch("veritatis.credibility._fetch_supabase_main_text_by_iids")
    @patch("veritatis.credibility.find_best_vector_with_embeddings")
    @patch("veritatis.credibility._get_record_embedding")
    @patch("veritatis.credibility.Collection")
    @patch("veritatis.credibility.ensure_collection_loaded")
    def test_compute_with_multiple_groups(
        self,
        mock_ensure_loaded,
        mock_collection_cls,
        mock_get_embedding,
        mock_find_best,
        mock_fetch_content,
        mock_get_store,
        mock_detector_class,
    ):
        """Test credibility computation with multiple similarity groups."""
        # Mock two similarity groups
        mock_group1 = SimilarityGroup(
            anchor_iid="iid1",
            anchor_content="Test content 1",
            similar_records=[
                {
                    "iid": "iid2",
                    "content": "Test content 2",
                    "credibility_score": 0.0,
                    "date": 0,
                    "domain": "test.com",
                    "similarity_score": 0.9,
                }
            ],
            similarity_threshold=0.85,
            group_size=2,
        )

        mock_group2 = SimilarityGroup(
            anchor_iid="iid4",
            anchor_content="Test content 4",
            similar_records=[
                {
                    "iid": "iid5",
                    "content": "Test content 5",
                    "credibility_score": 0.0,
                    "date": 0,
                    "domain": "test.com",
                    "similarity_score": 0.88,
                }
            ],
            similarity_threshold=0.85,
            group_size=2,
        )

        # Mock detector
        mock_detector = Mock()
        mock_detector.find_similar_groups.return_value = [mock_group1, mock_group2]
        mock_detector_class.return_value = mock_detector

        # Mock store with collection
        mock_store = Mock()
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection
        mock_store.update_credibility_scores.return_value = 6
        mock_get_store.return_value = mock_store

        embeddings = {f"iid{i}": np.random.rand(384).tolist() for i in range(1, 7)}

        def get_embedding_side_effect(collection, iid):
            return embeddings.get(iid)

        mock_get_embedding.side_effect = get_embedding_side_effect

        def mock_query(expr, output_fields, **kwargs):
            # Handle metadata queries
            for i in range(1, 7):
                iid = f"iid{i}"
                if iid in expr and "embedding" not in output_fields:
                    return [
                        {
                            "iid": iid,
                            "credibility_score": 0.0,
                            "date": 0,
                            "domain": "test.com",
                        }
                    ]

            # All records query
            if expr == "iid != ''":
                return [{"iid": f"iid{i}"} for i in range(1, 7)]
            return []

        mock_collection.query.side_effect = mock_query

        # Mock content fetching
        mock_fetch_content.return_value = {
            f"iid{i}": f"Test content {i}" for i in range(1, 7)
        }

        # Mock vector ranking - different scores for each group
        def mock_ranking_side_effect(vectors, **kwargs):
            iids = [v["iid"] for v in vectors]
            if "iid1" in iids:
                # First group
                a1 = Mock()
                a1.iid = "iid1"
                a1.combined_score = 0.9
                a2 = Mock()
                a2.iid = "iid2"
                a2.combined_score = 0.85
                return (a1, [a1, a2])
            else:
                # Second group
                a4 = Mock()
                a4.iid = "iid4"
                a4.combined_score = 0.75
                a5 = Mock()
                a5.iid = "iid5"
                a5.combined_score = 0.70
                return (a4, [a4, a5])

        mock_find_best.side_effect = mock_ranking_side_effect

        # Run computation
        stats = compute_credibility_scores(
            collection_name="test_collection",
            similarity_threshold=0.85,
        )

        # Assertions
        assert stats["total_records"] == 6
        assert stats["groups_found"] == 2
        assert stats["records_in_groups"] == 4
        assert stats["records_without_groups"] == 2
        assert stats["updated_count"] == 6

        # Verify updates
        updates = mock_store.update_credibility_scores.call_args[0][1]
        assert len(updates) == 6

        # Check grouped records preserve ordering and are calibrated
        updates_dict = {u["iid"]: u["credibility_score"] for u in updates}
        assert updates_dict["iid1"] > updates_dict["iid2"]
        assert updates_dict["iid4"] > updates_dict["iid5"]
        assert all(
            0.0 <= updates_dict[i] <= 0.95 for i in ("iid1", "iid2", "iid4", "iid5")
        )

        # Check ungrouped records have neutral score
        ungrouped = [iid for iid in ["iid3", "iid6"] if iid in updates_dict]
        for iid in ungrouped:
            assert updates_dict[iid] == 0.5


def test_calibrate_group_scores_prevents_saturation_for_small_group():
    """Two-item groups should not both map to perfect 1.0 scores."""
    a1 = Mock()
    a1.iid = "iid1"
    a1.combined_score = 1.0
    a2 = Mock()
    a2.iid = "iid2"
    a2.combined_score = 1.0

    scores = _calibrate_group_scores([a1, a2], group_size=2, neutral_score=0.5)

    assert scores["iid1"] == pytest.approx(scores["iid2"])
    assert 0.5 <= scores["iid1"] <= 0.95

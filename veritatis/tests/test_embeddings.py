"""Tests for the embedding generator."""

import math

from veritatis.embeddings import embedding_generator


def test_embedding_generator_returns_1024_dim_vector():
    """Embedding generator returns a 384-dim vector."""
    vec = embedding_generator.embed("Magnitude 5.2 earthquake near City X")
    assert isinstance(vec, list), "Embedding should be a list of floats"
    assert len(vec) == 384, f"Expected 1024-d vector, got {len(vec)}"


def test_embedding_is_normalized_close_to_one():
    """Embeddings are approximately unit-normalized."""
    vec = embedding_generator.embed("Another sample text about an earthquake event")
    norm = math.sqrt(sum(x * x for x in vec))
    assert 0.98 <= norm <= 1.02, f"Expected L2 norm ~1.0, got {norm}"

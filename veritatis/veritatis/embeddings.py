"""Text embedding generation using SentenceTransformers."""

import logging
import os
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingGenerator:
    """Generate text embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {model_name}...")
        logger.info("Downloading/loading model weights")
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        logger.info(f"Model {model_name} loaded successfully")

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        cleaned = text.strip()
        vec = self._model.encode(
            cleaned, show_progress_bar=False, normalize_embeddings=True
        )
        result: List[float] = vec.tolist()
        return result

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for multiple texts."""
        cleaned = [t.strip() for t in texts]
        vectors = self._model.encode(
            cleaned, show_progress_bar=False, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]


embedding_generator = EmbeddingGenerator()

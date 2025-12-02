import os
import threading
from typing import List

from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingGenerator:
	def __init__(self, model_name: str = _DEFAULT_MODEL):
		self._model_name = model_name
		self._lock = threading.Lock()
		self._model = SentenceTransformer(model_name)

	def embed(self, text: str) -> List[float]:
		cleaned = text.strip()
		with self._lock:
			vec = self._model.encode(cleaned, show_progress_bar=False, normalize_embeddings=True)
		return vec.tolist()

	def embed_batch(self, texts: List[str]) -> List[List[float]]:
		cleaned = [t.strip() for t in texts]
		with self._lock:
			vectors = self._model.encode(cleaned, show_progress_bar=False, normalize_embeddings=True)
		return [v.tolist() for v in vectors]

embedding_generator = EmbeddingGenerator()

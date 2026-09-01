from typing import List
import numpy as np
import re

class EmbeddingsEngine:
    """Lightweight local vector embedding engine for semantic search."""

    def __init__(self, vocab_size: int = 500):
        self.vocab = {}
        self.vocab_size = vocab_size

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def get_embedding(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for t in tokens:
            idx = hash(t) % self.vocab_size
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            return float(dot / (norm1 * norm2))
        return 0.0

embeddings_engine = EmbeddingsEngine()

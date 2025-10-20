import numpy as np
import hashlib

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

class SimpleEmbedder:
    """
    Lightweight, dependency-free embedder using a hashing trick to 384-dim vectors.
    Replace with a real model later (e.g., sentence-transformers).
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        text = (text or "").lower()
        vec = np.zeros(self.dim, dtype=float)
        for token in text.split():
            h = int(hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm


class STEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer not available")
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        v = self.model.encode(text or "", normalize_embeddings=True)
        return np.asarray(v, dtype=float)


def get_embedder():
    try:
        if SentenceTransformer is not None:
            return STEmbedder()
    except Exception:
        pass
    return SimpleEmbedder()

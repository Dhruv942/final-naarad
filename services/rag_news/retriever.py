import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def _encode_text(embedding_model, text: str) -> np.ndarray:
    try:
        vec = embedding_model.encode(text or "")
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm
    except Exception:
        return np.zeros(384, dtype=float)


def retrieve_candidates(articles: List[Dict], query: str, embedding_model, k: int = 50) -> List[Dict]:
    """
    Simple embedding-based retriever: scores by cosine similarity between query
    and concatenated title+summary+content, returns top-k.
    """
    try:
        qv = _encode_text(embedding_model, query)
        scored = []
        for a in articles:
            text = f"{a.get('title','')} {a.get('summary','')} {a.get('content','')}"
            av = _encode_text(embedding_model, text)
            sim = float(np.dot(qv, av))
            b = dict(a)
            b["retrieve_score"] = sim
            scored.append(b)
        scored.sort(key=lambda x: x.get("retrieve_score", 0.0), reverse=True)
        return scored[: max(k, 1)]
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        return articles[: max(k, 1)]

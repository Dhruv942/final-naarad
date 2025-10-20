import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def _encode(embedding_model, text: str) -> np.ndarray:
    try:
        v = embedding_model.encode(text or "")
        n = np.linalg.norm(v) or 1.0
        return v / n
    except Exception:
        return np.zeros(384, dtype=float)


def _intent_text(alert: Dict) -> str:
    fu = " ".join([(s or "") for s in (alert or {}).get("followup_questions", [])])
    cq = (alert or {}).get("custom_question", "") or ""
    return f"{fu} {cq}".strip()


def rerank_candidates(query: str, alert: Dict, candidates: List[Dict], embedding_model, n: int = 10) -> List[Dict]:
    """
    Lightweight reranker: combines retrieve_score with similarity to alert intent text.
    Replace with a cross-encoder later.
    """
    try:
        intent = _intent_text(alert)
        qv = _encode(embedding_model, query)
        iv = _encode(embedding_model, intent) if intent else None

        rescored: List[Dict] = []
        for c in candidates:
            text = f"{c.get('title','')} {c.get('summary','')} {c.get('content','')}"
            av = _encode(embedding_model, text)
            sem_sim = float(np.dot(qv, av))
            intent_sim = float(np.dot(iv, av)) if iv is not None else 0.0
            base = float(c.get("retrieve_score", sem_sim))
            score = 0.7 * base + 0.3 * intent_sim
            c2 = dict(c)
            c2["rerank_score"] = score
            c2["semantic_score"] = sem_sim
            c2["intent_score"] = intent_sim
            c2["_vec"] = av
            rescored.append(c2)
        rescored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return rescored[: max(n, 1)]
    except Exception as e:
        logger.error(f"Reranker failed: {e}")
        return candidates[: max(n, 1)]

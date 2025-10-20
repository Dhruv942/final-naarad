import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def mmr_diversify(items: List[Dict], vec_key: str = "_vec", score_key: str = "rerank_score", k: int = 10, alpha: float = 0.7) -> List[Dict]:
    """
    Maximal Marginal Relevance selection over items with vectors in vec_key and base scores in score_key.
    """
    if not items:
        return []
    n = min(len(items), max(k, 1))
    selected: List[Dict] = []
    selected_vecs: List[np.ndarray] = []
    candidates = list(range(len(items)))

    # Normalize base scores
    bases = [float(items[i].get(score_key, 0.0)) for i in range(len(items))]
    maxb = max(bases) if bases else 1.0
    maxb = maxb or 1.0
    bases = [b / maxb for b in bases]

    # Seed with the first (already sorted by rerank_score)
    first = candidates.pop(0)
    selected.append(items[first])
    v0 = items[first].get(vec_key)
    if isinstance(v0, np.ndarray):
        selected_vecs.append(v0)

    while candidates and len(selected) < n:
        best_idx = None
        best_score = -1e9
        for idx in candidates:
            v = items[idx].get(vec_key)
            if not isinstance(v, np.ndarray) or not selected_vecs:
                sim_pen = 0.0
            else:
                try:
                    sims = [float(np.dot(v, sv)) for sv in selected_vecs]
                    sim_pen = max(sims) if sims else 0.0
                except Exception:
                    sim_pen = 0.0
            mmr = alpha * bases[idx] - (1 - alpha) * sim_pen
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        if best_idx is None:
            break
        selected.append(items[best_idx])
        vb = items[best_idx].get(vec_key)
        if isinstance(vb, np.ndarray):
            selected_vecs.append(vb)
        candidates.remove(best_idx)

    return selected

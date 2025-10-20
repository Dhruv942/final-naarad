import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None


class CEReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder is not available")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, items: List[Dict], text_key: str = "title") -> List[Dict]:
        pairs = []
        idxs = []
        for i, it in enumerate(items):
            txt = f"{it.get('title','')} {it.get('summary','')} {it.get('content','')}"
            pairs.append((query, txt))
            idxs.append(i)
        scores = self.model.predict(pairs)
        out: List[Dict] = []
        for i, sc in zip(idxs, scores):
            c = dict(items[i])
            c["ce_score"] = float(sc)
            out.append(c)
        out.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
        return out


def get_ce_reranker() -> CEReranker | None:
    try:
        if CrossEncoder is not None:
            return CEReranker()
    except Exception:
        pass
    return None

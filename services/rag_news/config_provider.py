"""
Dynamic configuration provider for RAG News scoring.
Merges defaults -> per-category overrides (future) -> per-user/alert overrides.
Currently returns safe defaults with popularity_mode derived from alert.
"""
from __future__ import annotations

from typing import Any, Dict

from .config import (
    FRESH_TAU_BY_CATEGORY,
)


async def get_effective_config(category: str, user_id: str, alert: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve effective config for a scoring request.

    Merge order (future):
      defaults -> category_configs (Mongo) -> user/alert overrides

    For now, returns:
      - popularity_mode: from alert if present, else "boost"
      - fresh_tau: from FRESH_TAU_BY_CATEGORY
      - weights/thresholds/feature_flags: None (use module defaults)
    """
    # Category-aware freshness horizon
    fresh_tau = FRESH_TAU_BY_CATEGORY.get((category or "").lower(), 48)

    # Popularity preference from alert or default to boost
    popularity_mode = (alert or {}).get("popularity_mode", "boost")
    if popularity_mode not in ("off", "boost", "only_popular"):
        popularity_mode = "boost"

    return {
        "weights": None,            # use defaults in article_filter if None
        "thresholds": None,         # use defaults in article_filter if None
        "feature_flags": None,      # reserved
        "fresh_tau": fresh_tau,
        "popularity_mode": popularity_mode,
    }

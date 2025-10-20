import logging
from typing import Dict
from datetime import datetime, timezone
import numpy as np
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from .config import SOURCE_AUTHORITY, FRESH_TAU_BY_CATEGORY

logger = logging.getLogger(__name__)


def _source_authority_score(url: str) -> float:
    try:
        if not url:
            return SOURCE_AUTHORITY.get("default", 0.7)
        netloc = urlparse(url).netloc.lower()
        # exact domain or known keys
        for known in SOURCE_AUTHORITY.keys():
            if known in netloc:
                return SOURCE_AUTHORITY[known]
        # fallback strip to eTLD+1
        parts = netloc.split(":")[0].split(".")
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
        return SOURCE_AUTHORITY.get(domain, SOURCE_AUTHORITY.get("default", 0.7))
    except Exception:
        return SOURCE_AUTHORITY.get("default", 0.7)


def time_decay(published_date_str: str, category: str) -> float:
    try:
        tau_hours = FRESH_TAU_BY_CATEGORY.get((category or "").lower(), 48)
        tau = max(float(tau_hours), 1.0)
        if not published_date_str:
            return 0.9
        published_dt = parsedate_to_datetime(published_date_str)
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        age_hours = max((datetime.now(timezone.utc) - published_dt).total_seconds() / 3600.0, 0.0)
        return float(np.exp(-age_hours / tau))
    except Exception:
        return 0.8


def popularity_proxy(article: Dict, category: str) -> float:
    auth = _source_authority_score(article.get("url", ""))
    fresh = time_decay(article.get("published_date", ""), category)
    return float(0.6 * auth + 0.4 * fresh)

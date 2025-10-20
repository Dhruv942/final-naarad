from __future__ import annotations
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import math
import re


@dataclass
class UserProfile:
    user_id: str
    category: str
    sub_categories: List[str]
    followups: List[str]
    custom_question: str
    locales: List[str]
    preferred_sources: List[str]
    blocked_sources: List[str]
    recency_bias: float  # lambda for exp decay, e.g., 0.03
    diversity_bias: float  # 0..1 controls domain diversification strength
    personalization_weight: float  # 0..1 weight in final blend
    prefer_womens: bool  # user is explicitly interested in women's sports
    avoid_gossip: bool   # downweight gossip/prediction unless requested


def build_user_profile(alert: Dict[str, Any], user_id: str) -> UserProfile:
    subcats = [(s or "").strip().lower() for s in (alert.get("sub_categories") or [])]
    followups = [(s or "").strip().lower() for s in (alert.get("followup_questions") or [])]
    cq = (alert.get("custom_question") or "").strip().lower()
    # Women's interest if explicit markers present
    womens_markers = {"women", "womens", "women's", "w", "women cricket", "wpl"}
    prefer_womens = any(any(m in s for m in womens_markers) for s in (subcats + followups + [cq]))
    avoid_gossip = bool(alert.get("avoid_gossip", True))
    return UserProfile(
        user_id=user_id,
        category=(alert.get("main_category") or "").lower(),
        sub_categories=subcats,
        followups=followups,
        custom_question=cq,
        locales=[],
        preferred_sources=[s.lower() for s in (alert.get("preferred_sources") or [])],
        blocked_sources=[s.lower() for s in (alert.get("blocked_sources") or [])],
        recency_bias=float(alert.get("recency_bias", 0.03)),
        diversity_bias=float(alert.get("diversity_bias", 0.5)),
        personalization_weight=float(alert.get("personalization_weight", 0.4)),
        prefer_womens=prefer_womens,
        avoid_gossip=avoid_gossip,
    )


def _contains(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return any(n for n in needles if n and n in t)


def _contains_number(text: str, num: int) -> bool:
    if not text or num is None:
        return False
    pat = re.compile(r"\b" + re.escape(str(num)) + r"\b")
    return bool(pat.search(text))


def extract_features(article: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_domain": (article.get("source") or "").lower(),
        "title": article.get("title") or "",
        "description": article.get("description") or "",
        "content": article.get("content") or "",
        "published_ts": article.get("published_date") or "",
        "word_count": int(article.get("word_count") or 0),
        "category": (article.get("category") or "").lower(),
    }


# Category-specific keywords for boosts/penalties
CAT_BOOSTS: Dict[str, List[str]] = {
    "sports": ["odi", "test", "t20", "india vs", "ind vs", "match report", "post-match"],
    "news": ["breaking", "exclusive", "report", "analysis"],
    "business": ["stock", "stocks", "market", "nifty", "sensex", "ipos", "ipo", "results", "quarter", "q1", "q2", "q3", "q4", "guidance", "gold price", "gst"],
    "finance": ["stock", "sensex", "nifty", "ipo", "results", "earnings", "guidance", "fy25", "q2 results"],
    "tech": ["research", "paper", "arxiv", "github", "release", "patch", "security", "vulnerability"],
    "science": ["study", "research", "journal", "nature", "science", "peer-reviewed", "arxiv"],
    "entertainment": ["review", "box office", "trailer", "teaser", "release date"],
    "politics": ["election", "parliament", "policy", "bill", "supreme court", "cabinet"],
}

CAT_PENALTIES: Dict[str, List[str]] = {
    "sports": ["dream11", "fantasy", "prediction", "rumour", "rumor", "gossip"],
    "business": ["celebrity", "gossip", "viral", "dream11", "prediction"],
    "finance": ["celebrity", "gossip", "viral"],
    "entertainment": ["fake", "leak (unverified)", "rumour", "rumor"],
    "tech": ["giveaway", "contest", "celebrity"],
    "science": ["astrology", "horoscope"],
    "politics": ["gossip", "rumour", "rumor"],
}


def score_article(article: Dict[str, Any], profile: UserProfile, context_query: str) -> Tuple[float, Dict[str, float]]:
    f = extract_features(article)
    score = 0.0
    parts: Dict[str, float] = {}

    # Category/sub-category boost
    cat_boost = 0.2 if (profile.category and profile.category == f["category"]) else 0.0
    if cat_boost == 0.0 and profile.sub_categories:
        if _contains(f["title"] + " " + f["description"], profile.sub_categories):
            cat_boost = 0.15
    parts["category"] = cat_boost
    score += cat_boost

    # Followups/custom question boost
    q_text = (context_query or "") + " " + profile.custom_question
    fu_terms = [t for t in profile.followups if t]
    fu_boost = 0.0
    if fu_terms and _contains(f["title"] + " " + f["description"] + " " + f["content"], fu_terms):
        fu_boost += 0.35
    # Numeric constraint soft boost (e.g., "30 runs")
    nums = [int(m) for m in re.findall(r"\b\d+\b", profile.custom_question or "")]
    if nums:
        if any(_contains_number(f["title"] + " " + f["description"] + " " + f["content"], n) for n in nums):
            fu_boost += 0.2
    parts["followups_custom"] = fu_boost
    score += fu_boost

    # Named player/entity emphasis for cricket (common case): boost core names from followups/custom
    players = [p for p in ["rohit", "sharma", "kohli", "virat", "gill", "rahul", "rohit sharma", "virat kohli", "kl rahul", "hardik", "jadeja"] if p]
    if _contains(f["title"] + " " + f["description"] + " " + f["content"], players):
        score += 0.1
        parts["players"] = parts.get("players", 0.0) + 0.1

    # Preferred/blocked sources
    src = f["source_domain"]
    src_boost = 0.0
    if src:
        if profile.preferred_sources and any(ps in src for ps in profile.preferred_sources):
            src_boost += 0.1
        if profile.blocked_sources and any(bs in src for bs in profile.blocked_sources):
            src_boost -= 1.0  # strong penalty
    parts["source"] = src_boost
    score += src_boost

    # Content quality proxy
    wc = f["word_count"]
    qual = 0.0
    if wc >= 300:
        qual += 0.1
    elif wc == 0:
        qual -= 0.15
    parts["quality"] = qual
    score += qual

    # Avoid women's cricket unless explicitly requested
    text_all = (f["title"] + " " + f["description"] + " " + f["content"]).lower()
    womens_markers = [
        "women's", "womens", "women", "ind w", "eng w", "w vs", "w odi", "wpl", "women world cup",
    ]
    womens_flag = any(m in text_all for m in womens_markers)
    womens_adj = 0.0
    if f["category"] == "sports" and womens_flag and not profile.prefer_womens:
        womens_adj -= 0.6  # strong downweight if user didn't ask for it
    parts["women_filter"] = womens_adj
    score += womens_adj

    # Avoid gossip/prediction/fantasy unless requested
    gossip_terms = [
        "gossip", "rumour", "rumor", "dating", "affair", "controversy", "trolled", "viral",
        "dream11", "fantasy", "prediction", "who will win", "astrology", "horoscope"
    ]
    gossip_adj = 0.0
    if profile.avoid_gossip and _contains(text_all, gossip_terms):
        gossip_adj -= 0.5
    parts["gossip_filter"] = gossip_adj
    score += gossip_adj

    # Live streaming/how-to-watch penalty for sports unless explicitly requested
    wants_live = (
        _contains((profile.custom_question or ""), ["live", "stream", "telecast", "watch"]) or
        _contains(" ".join(profile.followups), ["live", "stream", "telecast", "watch"])
    )
    live_terms = [
        "live streaming", "live telecast", "how to watch", "where to watch", "watch live", "live blog", "live score"
    ]
    live_adj = 0.0
    if f["category"] == "sports" and _contains(text_all, live_terms) and not wants_live:
        live_adj -= 0.4
    parts["live_filter"] = live_adj
    score += live_adj

    # Category-specific boosts and penalties
    cat = f["category"]
    cat_adj = 0.0
    if cat in CAT_BOOSTS:
        if _contains(text_all, CAT_BOOSTS[cat]):
            cat_adj += 0.15
    if cat in CAT_PENALTIES:
        if _contains(text_all, CAT_PENALTIES[cat]):
            cat_adj -= 0.2
    parts["category_specific"] = cat_adj
    score += cat_adj

    return score, parts


def rerank_articles(articles: List[Dict[str, Any]], profile: UserProfile, context_query: str) -> List[Dict[str, Any]]:
    # Compute personalized scores
    rescored = []
    for it in articles:
        pscore, parts = score_article(it, profile, context_query)
        base = float(it.get("final_score", it.get("rerank_score", it.get("retrieve_score", 0.0))))
        w = max(0.0, min(1.0, profile.personalization_weight))
        final = (1 - w) * base + (w) * pscore
        it2 = dict(it)
        it2["personalization_score"] = pscore
        it2["personalization_breakdown"] = parts
        it2["final_score"] = final
        rescored.append(it2)

    # Domain diversity: penalize repeated domains progressively
    domain_counts = {}
    def domain_penalty(domain: str, idx: int) -> float:
        c = domain_counts.get(domain, 0)
        domain_counts[domain] = c + 1
        if c == 0:
            return 0.0
        return profile.diversity_bias * math.log1p(c)

    rescored.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    adjusted = []
    for idx, it in enumerate(rescored):
        dom = (it.get("source") or it.get("source_domain") or "").lower()
        pen = domain_penalty(dom, idx)
        it2 = dict(it)
        it2["final_score"] = it2.get("final_score", 0.0) - pen
        adjusted.append(it2)

    adjusted.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return adjusted


def enforce_must_haves(candidates: List[Dict[str, Any]], profile: UserProfile) -> List[Dict[str, Any]]:
    # If custom question has specific numbers/entities, ensure at least one top result matches strictly
    if not profile.custom_question:
        return candidates
    terms = [t for t in re.findall(r"[a-zA-Z]{3,}", profile.custom_question) if t]
    nums = [int(m) for m in re.findall(r"\b\d+\b", profile.custom_question)]
    for i, it in enumerate(candidates[:10]):
        text = (it.get("title") or "") + " " + (it.get("description") or "") + " " + (it.get("content") or "")
        if terms and not _contains(text, [t.lower() for t in terms]):
            continue
        if nums and not any(_contains_number(text, n) for n in nums):
            continue
        # Found a strict match; promote to position 1
        if i != 0:
            candidates.insert(0, candidates.pop(i))
        return candidates
    return candidates
